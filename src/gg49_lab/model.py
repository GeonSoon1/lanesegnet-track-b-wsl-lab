import sys,random,json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config,ConfigDict
from mmdet.models.builder import LOSSES,build_detector
from .common import ROOT,STORE,write_json
@LOSSES.register_module(force=True)
class MaskedLaneTypeLoss(nn.Module):
    def __init__(self,loss_weight=.1,**kwargs):super().__init__();self.loss_weight=loss_weight
    def forward(self,pred,target,weight=None,avg_factor=None,**kwargs):
        valid=(target>=0)&(target<3)
        if not valid.any():return pred.sum()*0
        loss=F.cross_entropy(pred[valid],target[valid].long(),reduction='none')
        if weight is not None:loss=loss*weight[valid]
        return loss.mean()*self.loss_weight
class LiDARFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(5,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU())
        self.projection=nn.Conv2d(64,256,1);nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
    def forward(self,bev,lidar):
        x=self.projection(self.encoder(lidar.float()))*lidar[:,4:5]
        return bev+x.flatten(2).transpose(1,2)
class LabModel(nn.Module):
    def __init__(self,mode='camera',load_initial=True):
        super().__init__();sys.path.insert(0,str(ROOT/'engine'))
        import projects.bevformer,projects.lanesegnet
        cfg=Config.fromfile(str(ROOT/'engine/projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py'))
        cfg.model.img_backbone.init_cfg=None;cfg.model.img_backbone.with_cp=True
        cfg.model.bev_constructor.num_cams=6
        cfg.model.bev_constructor.encoder.transformerlayers.attn_cfgs[1].num_cams=6
        def update(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in ('pc_range','point_cloud_range'):x[k]=[-51.2,-25.6,-4,51.2,25.6,2]
                    else:update(v)
            elif isinstance(x,(list,tuple)):
                for v in x:update(v)
        update(cfg.model);cfg.model.lane_head.num_classes=1
        cfg.model.lane_head.loss_lane_type=ConfigDict(type='MaskedLaneTypeLoss',use_sigmoid=True,loss_weight=.1)
        self.net=build_detector(cfg.model);self.net.init_weights();self.net.pts_bbox_head.align_inference_reference=True;self.mode=mode;self.use_lidar=bool(json.loads((ROOT/'configs/model_modes.json').read_text())[mode]['use_lidar'])
        if load_initial:
            path=STORE/'checkpoints/lanesegnet_r50_8x1_24e_olv2_subset_A.pth'
            state=torch.load(path,map_location='cpu',weights_only=True)['state_dict'];own=self.net.state_dict()
            accepted={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
            report={'initial_checkpoint':str(path),'loaded':list(accepted),'excluded':{k:{'source':list(v.shape),'target':list(own[k].shape) if k in own else None} for k,v in state.items() if k not in accepted},'new':list(set(own)-set(accepted))}
            self.net.load_state_dict(accepted,strict=False);nn.init.normal_(self.net.bev_constructor.cams_embeds,std=.02)
            write_json(STORE/'reports'/('checkpoint-load-'+mode+'-'+str(__import__('os').getpid())+'.json'),report)
        self.fusion=LiDARFusion() if self.use_lidar else None
    def features(self,img,metas,lidar=None):
        feats=self.net.extract_feat(img=img,img_metas=metas)
        # Image CNN uses AMP; geometric attention/head and assignment need FP32.
        with torch.cuda.amp.autocast(enabled=False):
            feats=[x.float() for x in feats]
            bev=self.net.bev_constructor(feats,metas,None)
            if self.fusion is not None:
                if lidar is None:raise ValueError('Fusion requires lidar_bev')
                bev=self.fusion(bev,lidar.float())
        return feats,bev
    def forward(self,img,metas,targets,lidar=None):
        feats,bev=self.features(img,metas,lidar)
        with torch.cuda.amp.autocast(enabled=False):
            head=self.net.pts_bbox_head
            previous=getattr(head,'compute_eval_loss',False);head.compute_eval_loss=True
            try:outs=head(feats,bev,metas)
            finally:head.compute_eval_loss=previous
            for key,value in outs.items():
                if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise FloatingPointError('Non-finite head output: '+key)
            loss,assignment=self.net.pts_bbox_head.loss(outs,targets['lanes'],targets['labels'],targets['masks'],targets['left_types'],targets['right_types'],img_metas=metas)
            result={'lane_head.'+k:v for k,v in loss.items()}
            rel=self.net.lclc_head.forward_train(outs['history_states'],assignment,outs['history_states'],assignment,targets['adjacency'])
            result.update({'topology.'+k:v for k,v in rel.items()});return result
    @torch.no_grad()
    def predict(self,img,metas,lidar=None):
        feats,bev=self.features(img,metas,lidar)
        with torch.cuda.amp.autocast(enabled=False):
            outs=self.net.pts_bbox_head(feats,bev,metas)
            return self.net.pts_bbox_head.get_lanes(outs,metas),self.net.lclc_head.get_relationship(outs['history_states'],outs['history_states'])

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
