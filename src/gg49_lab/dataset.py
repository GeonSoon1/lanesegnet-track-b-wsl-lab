import math
import numpy as np,torch,cv2
from .common import STORE,read_json,ensure_storage
MEAN=np.array([123.675,116.28,103.53],np.float32);STD=np.array([58.395,57.12,57.375],np.float32)
def load_frames():
    ensure_storage()
    path=STORE/'artifacts/prepared/DATASET.json'
    data=read_json(path)
    if not isinstance(data,dict) or not isinstance(data.get('frames'),list) or not data['frames']:
        raise RuntimeError(f'Missing or invalid prepared dataset: {path}; connect/mount the experiment D: drive first')
    return data['frames']
class Dataset:
    def __init__(self,resolution=384):self.frames=load_frames();self.resolution=resolution
    def get(self,idx,device='cuda'):
        row=self.frames[idx];size=self.resolution
        with np.load(STORE/'cache/rgb'/(row['key']+'.npz'),allow_pickle=False) as d:rgb=d['rgb'];mats=d['matrices'].copy()
        if size!=384:rgb=np.stack([cv2.resize(a,(size,size),interpolation=cv2.INTER_AREA) for a in rgb]);mats[:,:2,:]*=size/384
        img=torch.from_numpy(((rgb.astype(np.float32)-MEAN)/STD).transpose(0,3,1,2).copy()).unsqueeze(0).to(device)
        R=np.array(row['rotation'],np.float32);yaw=math.atan2(R[1,0],R[0,0]);bus=np.zeros(18,np.float32);bus[-2:]=(yaw,math.degrees(yaw)%360);bus[3]=math.cos(yaw/2);bus[6]=math.sin(yaw/2)
        shapes=[(size,size,3)]*6
        meta={'filename':row['camera_files'],'ori_shape':shapes,'img_shape':shapes,'pad_shape':shapes,'lidar2img':list(mats),'img_norm_cfg':{'mean':MEAN,'std':STD,'to_rgb':True},'sample_idx':row['key'],'scene_token':'B','frame_idx':row['frame'],'timestamp':row['timestamp'],'can_bus':bus,'lidar2global_rotation':R,'flip':False,'scale_factor':size/2046,'prev_bev':False}
        with np.load(row['gt_path'],allow_pickle=False) as d:
            n=len(d['lanes']);targets={k:[torch.from_numpy(d[k].copy()).to(device)] for k in ['labels','masks','left_types','right_types','adjacency']};targets['lanes']=[torch.from_numpy(d['lanes'].reshape(n,90).copy()).to(device)]
        with np.load(STORE/'cache/lidar'/(row['key']+'.npz'),allow_pickle=False) as d:lidar=torch.from_numpy(d['bev'].astype(np.float32)).unsqueeze(0).to(device)
        return img,[meta],targets,lidar
    def batch(self,ids,device='cuda'):
        a=[self.get(i,device) for i in ids]
        return torch.cat([s[0] for s in a]),sum([s[1] for s in a],[]),{k:sum([s[2][k] for s in a],[]) for k in a[0][2]},torch.cat([s[3] for s in a])
