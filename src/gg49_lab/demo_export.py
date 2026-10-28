"""Export a completed demonstration checkpoint's own validation predictions to SHP."""
import argparse,copy,hashlib,json,logging,zipfile
from pathlib import Path
import numpy as np
import torch,fiona
from torch.cuda.amp import autocast
from .common import STORE,ensure_storage,read_json,write_json,now
from .dataset import Dataset
from .model import LabModel,seed_all
from .diagnostics import gpu_check
from .map_experiment import gt_reference,raw_links

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=ap.parse_args()
 if Path(args.run).name!=args.run:ap.error('run must be a folder name')
 ensure_storage();run=STORE/'runs'/args.run;summary=read_json(run/'summary.json')
 assert summary.get('status')=='completed' and summary['settings']['mode']=='camera'
 ckpath=run/'best.pth';ckhash=hashlib.sha256(ckpath.read_bytes()).hexdigest()
 dataset_hash=hashlib.sha256((STORE/'artifacts/prepared/DATASET.json').read_bytes()).hexdigest()
 assert dataset_hash==summary['settings']['dataset_sha256']
 out=run/'qgis-export-v1'
 if out.exists():raise RuntimeError('Preserve existing export; choose another version to rerun')
 if args.device=='cuda':gpu_check()
 torch.set_num_threads(2);seed_all();logging.disable(logging.INFO)
 model=LabModel('camera',load_initial=False).to(args.device)
 ck=torch.load(ckpath,map_location='cpu');model.load_state_dict(ck['model'],strict=True)
 epoch=ck['epoch'];del ck;model.eval()
 # Match the demonstration preview's default inference path; no reference experiment override.
 dataset=Dataset(summary['settings']['resolution']);indices=[i for i,r in enumerate(dataset.frames) if r['split']=='val'][:20]
 rows=[dataset.frames[i] for i in indices];out.mkdir();(out/'predictions').mkdir()
 features=[];counts=[];peak=0
 for idx,row in zip(indices,rows):
  img,meta,target,lidar=dataset.get(idx,device=args.device)
  with torch.no_grad(),autocast():pred,_=model.predict(img.clone(),copy.deepcopy(meta))
  p=pred[0];lanes=np.asarray(p[0],dtype=np.float32).reshape(-1,3,10,3);scores=np.asarray(p[1],dtype=np.float32)
  left=np.asarray(p[4]);right=np.asarray(p[6]);np.savez_compressed(out/'predictions'/(row['key']+'.npz'),lanes=lanes,scores=scores,left_types=left,right_types=right)
  selected=np.flatnonzero(scores>=.5);R=np.asarray(row['rotation']);origin=np.asarray(row['origin'])
  for rank in selected:
   features.append({'xyz':lanes[rank]@R.T+origin,'properties':{'id':f'{row["key"]}:{rank}','frame':row['key'],'score':float(scores[rank]),'left_type':int(left[rank]),'right_type':int(right[rank])}})
  counts.append({'frame':row['key'],'predictions_score_ge_05':len(selected),'max_score':float(scores.max()) if len(scores) else None})
  peak=max(peak,gpu_check()) if args.device=='cuda' else 0;print('INFERENCE',row['key'],'selected',len(selected),flush=True)
  del img,meta,target,lidar
 del model;torch.cuda.empty_cache()
 truth,_,_=gt_reference(rows,raw_links())
 layers={}
 for name,part in [('pred_centers',0),('pred_left',1),('pred_right',2)]:
  layers[name]=[{'geometry':{'type':'LineString','coordinates':r['xyz'][part].tolist()},'properties':r['properties']} for r in features]
 layers['gt_reference']=[{'geometry':{'type':'LineString','coordinates':r['xyz'].tolist()},'properties':{'id':r['id']}} for r in truth]
 verified=[]
 for name,items in layers.items():
  props={'id':'str:80'} if name=='gt_reference' else {'id':'str:80','frame':'str:20','score':'float:24.15','left_type':'int:9','right_type':'int:9'}
  path=out/(name+'.shp')
  with fiona.open(path,'w',driver='ESRI Shapefile',crs='EPSG:32652',schema={'geometry':'3D LineString','properties':props},encoding='UTF-8') as sink:
   for item in items:sink.write(item)
  with fiona.open(path) as src:
   assert len(src)==len(items) and src.crs.to_epsg()==32652 and src.schema['geometry']=='3D LineString'
   for a,b in zip(src,items):
    np.testing.assert_allclose(np.asarray(a['geometry']['coordinates']),b['geometry']['coordinates'],rtol=0,atol=1e-9)
    for k,v in b['properties'].items():
     if isinstance(v,float):assert abs(a['properties'][k]-v)<1e-10
     else:assert a['properties'][k]==v
  verified.append({'layer':name,'features':len(items),'xyz_and_attributes_verified':True})
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(13,5));allxyz=np.concatenate([r['xyz'] for r in truth]+[r['xyz'][0] for r in features]);origin=allxyz[:,:2].mean(0)
 for j,r in enumerate(truth):ax.plot(r['xyz'][:,0]-origin[0],r['xyz'][:,1]-origin[1],color='#8b5ca6',ls='--',lw=1.6,label='SHP ground truth' if j==0 else None)
 for j,r in enumerate(features):ax.plot(r['xyz'][0,:,0]-origin[0],r['xyz'][0,:,1]-origin[1],color='#247fb3',alpha=.45,lw=1,label='1-epoch camera predictions' if j==0 else None)
 ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Easting offset (m)');ax.set_ylabel('Northing offset (m)');ax.set_title(f'{args.run} | best epoch {epoch} | {rows[0]["key"]}..{rows[-1]["key"]}\nscore >= 0.5; {len(features)} predictions; NOT stitched; EPSG:32652')
 fig.tight_layout();fig.savefig(out/'preview.png',dpi=160);plt.close(fig)
 report={'status':'completed','completed_at':now(),'run':args.run,'checkpoint':str(ckpath),'checkpoint_sha256':ckhash,'checkpoint_epoch':epoch,'dataset_sha256':dataset_hash,'mode':'camera','device':args.device,'reference':'aligned last reference, same as this new demo preview','split':'val','frame_keys':[r['key'] for r in rows],'confidence':.5,'crs':'EPSG:32652','stitched':False,'prediction_count':len(features),'counts':counts,'gpu_peak_observed_mib':peak,'verified_layers':verified}
 write_json(out/'export-summary.json',report)
 (out/'README_KO.md').write_text(f"# 방금 학습한 1 epoch 모델의 QGIS 결과\n\n학습: {args.run}, best.pth epoch {epoch}. 영상 전용 모델입니다. 기존 24 epoch 모델은 사용하지 않았습니다.\n\n첫 validation 20시점 {rows[0]['key']}~{rows[-1]['key']}를 이번 가중치로 실제 추론했습니다. confidence 0.5 이상 예측 {len(features)}개. 정합 전 결과이므로 같은 차선이 여러 줄로 보일 수 있습니다.\n\nQGIS에 pred_centers.shp와 gt_reference.shp를 끌어 넣고 레이어로 확대하세요. 예측 파랑, 정답 보라로 비교하세요. pred_left.shp/pred_right.shp는 예측 좌우 경계입니다. 속성 frame을 기준으로 한 시점만 필터할 수 있습니다. 좌표계 EPSG:32652, XYZ 보존. 같은 이름의 shp/shx/dbf/prj/cpg를 함께 보관하세요.\n\n예측이 없거나 정답에서 멀어도 결과 그대로 저장합니다. 1 epoch 시연이며 성능 개선이나 지도 완성을 뜻하지 않습니다. GT는 비교에만 사용했습니다. 추가 학습이나 병합은 하지 않았습니다.\n",encoding='utf-8')
 assert ckhash==hashlib.sha256(ckpath.read_bytes()).hexdigest()
 manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()};write_json(out/'manifest.json',manifest)
 with zipfile.ZipFile(run/'qgis-export-v1.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,str(p.relative_to(run)))
 with zipfile.ZipFile(run/'qgis-export-v1.zip') as z:assert z.testzip() is None
 print('DEMO_SHP_EXPORT_COMPLETE',out, 'predictions',len(features),flush=True)
if __name__=='__main__':main()
