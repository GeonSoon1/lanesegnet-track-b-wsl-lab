"""Controlled decoder-reference experiment. Fixed weights and validation only."""
import copy,hashlib,json,logging,os,traceback
from pathlib import Path
import numpy as np,torch
from torch.cuda.amp import autocast
from .common import STORE,ROOT,read_json,write_json,now,ensure_storage
from .dataset import Dataset
from .model import LabModel,seed_all
from .diagnostics import load_gt,gpu_check
from .diagnostic_metrics import measure,combine,direction_free_geometry
OUT=STORE/'reports/followup-45-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(phase,done=0,total=94,**extra):
 write_json(STORE/'runs/followup-status.json',dict(status='running',phase=phase,completed=done,total=total,updated_at=now(),pid=os.getpid(),**extra));print(phase,done,total,extra,flush=True)

def main():
 ensure_storage();torch.set_num_threads(4);dataset=Dataset();protocol=read_json(OUT/'protocol.json');rows=[r for r in dataset.frames if r['key'] in protocol['frame_keys']];assert len(rows)==47 and all(r['split']=='val' for r in rows)
 provenance={'dataset_sha256':sha(STORE/'artifacts/prepared/DATASET.json'),'checkpoint_sha256':{m:sha(STORE/'runs'/f'{m}-24e/best.pth') for m in ['camera','fusion']},'reference_flag':'opt-in align_inference_reference, legacy default retained'}
 provenance['inference_source_sha256']={p:sha(ROOT/p) for p in ['src/gg49_lab/model.py','src/gg49_lab/dataset.py','engine/projects/lanesegnet/models/dense_heads/laneseg_head.py','engine/projects/lanesegnet/models/modules/laneseg_decoder.py']}
 previous=read_json(OUT/'reference-results.json')
 if previous and previous['provenance']!=provenance:raise RuntimeError('Inference/data/checkpoint provenance changed; use a new output version, not existing caches')
 assert provenance['dataset_sha256']==protocol['dataset_sha256']
 done=0;proofs=[];peak=0
 for mode in ['camera','fusion']:
  seed_all();logging.disable(logging.INFO)
  try:model=LabModel(mode,load_initial=False).cuda()
  finally:logging.disable(logging.NOTSET)
  ck=torch.load(STORE/'runs'/f'{mode}-24e/best.pth',map_location='cpu');model.load_state_dict(ck['model']);assert ck['epoch']==6;del ck;model.eval();head=model.net.pts_bbox_head
  captured={}
  def hook(module,args,output):
   captured['raw_lanes']=output['all_lanes_preds'][-1,0].detach().cpu().numpy().reshape(-1,3,10,3)
   captured['raw_scores']=output['all_cls_scores'][-1,0].sigmoid().detach().cpu().numpy().reshape(-1)
  h=head.register_forward_hook(hook)
  def predict(img,metas,lidar,aligned,full=False):
   head.align_inference_reference=aligned;head.compute_eval_loss=full;captured.clear()
   try:
    with torch.no_grad(),autocast():pred,_=model.predict(img.clone(),copy.deepcopy(metas),lidar if mode=='fusion' else None)
   finally:head.compute_eval_loss=False
   p=pred[0];return dict(lanes=np.asarray(p[0],np.float32).reshape(-1,3,10,3),scores=np.asarray(p[1],np.float32),left_types=np.asarray(p[4]),right_types=np.asarray(p[6]),**{k:v.copy() for k,v in captured.items()})
  for row in rows:
   key=row['key'];idx=next(i for i,r in enumerate(dataset.frames) if r['key']==key);dest=OUT/'predictions'/f'{mode}_aligned'/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   old=STORE/'reports/diagnostics-track-b-v1/predictions'/('camera' if mode=='camera' else 'fusion_normal')/(key+'.npz')
   with np.load(old) as a:legacy={k:a[k].copy() for k in a.files}
   img,metas,target,lidar=dataset.get(idx)
   if dest.exists():
    with np.load(dest) as a:aligned={k:a[k].copy() for k in a.files}
   else:
    aligned=predict(img,metas,lidar,True);np.savez_compressed(dest,**aligned)
   assert np.max(np.abs(aligned['raw_scores']-legacy['raw_scores']))<1e-6
   if key in protocol['full_path_equivalence_keys']:
    current_legacy=predict(img,metas,lidar,False);full=predict(img,metas,lidar,True,True)
    p={'mode':mode,'key':key,'legacy_cache_max_xyz_delta_m':float(np.max(np.abs(current_legacy['raw_lanes']-legacy['raw_lanes']))),'aligned_vs_training_path_max_xyz_delta_m':float(np.max(np.abs(full['raw_lanes']-aligned['raw_lanes']))),'aligned_vs_training_path_max_score_delta':float(np.max(np.abs(full['raw_scores']-aligned['raw_scores'])))}
    assert p['legacy_cache_max_xyz_delta_m']<1e-5 and p['aligned_vs_training_path_max_xyz_delta_m']<1e-5 and p['aligned_vs_training_path_max_score_delta']<1e-6,p
    proofs.append(p)
   done+=1;peak=max(peak,gpu_check());state('reference_inference',done,94,mode=mode,frame=key,gpu_peak_observed_mib=peak)
   del img,metas,target,lidar
  h.remove();del model;torch.cuda.empty_cache()
 result={'status':'completed','completed_at':now(),'protocol':protocol,'provenance':provenance,'equivalence':proofs,'gpu_peak_observed_mib':peak,'variants':{},'cases':[]}
 preds={}
 for variant in ['camera_legacy','camera_aligned','fusion_legacy','fusion_aligned']:
  preds[variant]={}
  for row in rows:
   if variant.endswith('legacy'):p=STORE/'reports/diagnostics-track-b-v1/predictions'/('camera' if variant.startswith('camera') else 'fusion_normal')/(row['key']+'.npz')
   else:p=OUT/'predictions'/variant/(row['key']+'.npz')
   with np.load(p) as a:preds[variant][row['key']]={k:a[k].copy() for k in a.files}
  metrics={};prox=[]
  for tolerance in [.5,1.,2.]:metrics[str(tolerance)+'m']=combine([measure(preds[variant][r['key']]['lanes'],preds[variant][r['key']]['scores'],load_gt(r),.5,tolerance) for r in rows])
  for row in rows:
   a=preds[variant][row['key']];prox.append(direction_free_geometry(a['lanes'][a['scores']>=.5],load_gt(row)))
  gl=sum(p['gt_length_m'] for p in prox);pl=sum(p['pred_length_m'] for p in prox)
  result['variants'][variant]={'geometry':metrics,'proximity':{'gt_supported_fraction':sum(p['gt_supported_length_m'] for p in prox)/gl,'pred_supported_fraction':sum(p['pred_supported_length_m'] for p in prox)/pl if pl else None}}
 for row in rows:
  case={'key':row['key'],'metrics':{},'lengths_m':{},'gt_lengths_m':np.linalg.norm(np.diff(load_gt(row)[:,0],axis=1),axis=-1).sum(-1).tolist()}
  for v in preds:
   a=preds[v][row['key']];selected=a['lanes'][a['scores']>=.5];case['metrics'][v]=measure(a['lanes'],a['scores'],load_gt(row),.5,1.);case['lengths_m'][v]=np.linalg.norm(np.diff(selected[:,0],axis=1),axis=-1).sum(-1).tolist()
  result['cases'].append(case)
 assert provenance['checkpoint_sha256']=={m:sha(STORE/'runs'/f'{m}-24e/best.pth') for m in ['camera','fusion']}
 write_json(OUT/'reference-results.json',result);state('reference_completed',94,94)
 print('REFERENCE_EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(STORE/'runs/followup-status.json',{'status':'failed','phase':'reference','error':str(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
