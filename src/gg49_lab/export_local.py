"""Export this run's measured held-out inference; keep raw and stitched layers separate."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np,fiona,torch
from .common import STORE,read_json,write_json,ensure_storage,now,status,STORAGE
from .dataset import load_frames
from .map_experiment import gt_reference,raw_links
from .stitching import stitch,geometry_quality
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def finalize_if_ready():
 if all((STORE/'runs'/(STORAGE.get('run_prefix','')+m+'-24e')/'qgis-export-v1.zip').is_file() for m in ['camera','fusion']):
  from .finalize_local import main as finalize
  finalize()
  status('done','completed','Both new model evaluations and verified QGIS SHP ZIP exports completed',share_zip='exports/'+STORAGE.get('run_prefix','')+'qgis-comparison.zip')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();ensure_storage()
 if Path(a.run).name!=a.run:raise ValueError('Invalid run name')
 run=STORE/'runs'/a.run;summary=read_json(run/'summary.json');assert summary['status']=='completed'
 cp=run/'best.pth';ck=torch.load(cp,map_location='cpu');assert ck['settings']['dataset_sha256']==sha(STORE/'artifacts/prepared/DATASET.json')
 epoch=ck['epoch'];del ck
 world=read_json(run/'world-predictions.json')
 assert world['provenance']['checkpoint_sha256']==sha(cp),'Inference belongs to a different checkpoint'
 assert world['provenance']['dataset_sha256']==summary['settings']['dataset_sha256']
 frames=[x for x in world['frames'] if x['split']=='test']
 assert frames,'Held-out inference missing'
 out=run/'qgis-export-v1'
 if out.exists():
  assert read_json(out/'export-summary.json').get('checkpoint_sha256')==sha(cp),'Existing export has different provenance'
  with zipfile.ZipFile(run/'qgis-export-v1.zip') as z:assert z.testzip() is None
  finalize_if_ready()
  print('Existing verified export preserved',out);return
 out.mkdir();observations=[]
 for row in frames:
  for i,(lane,score) in enumerate(zip(row['lanes_center_left_right_xyz'],row['scores'])):
   observations.append(dict(id=f"{row['key']}:{i}",frame=row['key'],timestamp=row['timestamp'],lanes=np.asarray(lane),score=score,left_type=row['left_types'][i],right_type=row['right_types'][i]))
 keys={r['key'] for r in frames};rows=[r for r in load_frames() if r['key'] in keys]
 truth,_,gt_audit=gt_reference(rows,raw_links())
 write_json(out/'gt-reference-audit.json',{'source':'Provided NT2_LINK clipped to selected test-frame ROI union; source IDs taken from training-GT provenance','gt_instance_to_source_distance':gt_audit,'not_prediction':True})
 protocol=dict(max_angle_deg=15,median_distance_m=.65,p95_distance_m=1.,max_z_m=.5,min_overlap_m=5.,min_overlap_ratio=.25,station_step_m=1.,max_unobserved_gap_m=1.5,min_length_m=5.)
 merged,ledger=stitch(observations,protocol)
 layers={}
 for prefix,items in [('pred',observations),('merged',merged)]:
  for name,part in [('centers',0),('left',1),('right',2)]:
   layers[prefix+'_'+name]=[dict(xyz=o['lanes'][part],id=o['id'],frame=o.get('frame','multiple'),score=o['score'],ltype=o['left_type'],rtype=o['right_type'],n_src=o.get('source_count',1)) for o in items]
 layers['gt_reference']=[dict(xyz=o['xyz'],id=o['id'],frame='reference',score=-1.,ltype=-1,rtype=-1,n_src=0) for o in truth]
 verified=[]
 for name,items in layers.items():
  fields=['id','frame','score','ltype','rtype','n_src']
  path=out/(name+'.shp');schema={'geometry':'3D LineString','properties':{'id':'str:100','frame':'str:30','score':'float:24.15','ltype':'int:9','rtype':'int:9','n_src':'int:9'}}
  with fiona.open(path,'w',driver='ESRI Shapefile',crs='EPSG:32652',schema=schema,encoding='UTF-8') as sink:
   for o in items:sink.write({'geometry':{'type':'LineString','coordinates':o['xyz'].tolist()},'properties':{k:o[k] for k in fields}})
  with fiona.open(path) as source:
   assert len(source)==len(items) and source.crs.to_epsg()==32652 and source.schema['geometry']=='3D LineString'
   for x,y in zip(source,items):
    np.testing.assert_allclose(x['geometry']['coordinates'],y['xyz'],atol=1e-9,rtol=0)
    assert all(x['properties'][k]==y[k] for k in fields if k!='score')
    assert abs(x['properties']['score']-y['score'])<1e-10
  assert all(path.with_suffix(e).exists() for e in ['.shp','.shx','.dbf','.prj','.cpg'])
  verified.append({'layer':name,'count':len(items),'CRS_XYZ_properties_verified':True})
 write_json(out/'association-ledger.json',ledger)
 write_json(out/'merge-protocol.json',protocol)
 # Source observations remain available even if association rejected them.
 write_json(out/'provenance.json',{'observations':[dict(o,lanes=o['lanes'].tolist()) for o in observations],'merged':[dict(o,lanes=o['lanes'].tolist()) for o in merged]})
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,items,title in zip(axes,[observations,merged],['Raw per-frame predictions','GT-free stitched predictions']):
  for o in truth:ax.plot(o['xyz'][:,0],o['xyz'][:,1],color='purple',ls='--',lw=.8)
  for o in items:ax.plot(o['lanes'][0,:,0],o['lanes'][0,:,1],color='steelblue',alpha=.6,lw=.8)
  ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('EPSG:32652 Easting (m)');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'preview.png',dpi=150,bbox_inches='tight');plt.close(fig)
 report={'status':'completed','time':now(),'run':a.run,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'checkpoint_epoch':epoch,'dataset_sha256':sha(STORE/'artifacts/prepared/DATASET.json'),'split':'test','frame_keys':sorted(keys),'confidence':.5,'CRS':'EPSG:32652','inference':'fresh run evaluation outputs, aligned last reference','layers':verified,'raw_count':len(observations),'merged_count':len(merged),'merge_evaluation':'Postprocessing only; fewer lines do not establish improved accuracy.'}
 write_json(out/'export-summary.json',report)
 quality={'method':'Length-weighted sampled nearest-segment XYZ distance at1m; custom map diagnostic, not OpenLane-V2 benchmark','raw':geometry_quality([o['lanes'][0] for o in observations],[o['xyz'] for o in truth]),'merged':geometry_quality([o['lanes'][0] for o in merged],[o['xyz'] for o in truth])}
 write_json(out/'map-quality-comparison.json',quality)
 (out/'README_KO.txt').write_text('이번 PC에서 새로 학습한 '+a.run+' best checkpoint의 test 추론 결과입니다.\nQGIS에 pred_centers.shp(예측 중심선), pred_left.shp와 pred_right.shp(예측 경계), gt_reference.shp(비교 정답)를 함께 여세요.\nmerged_*는 같은 예측을 GT 없이 대응·가중평균한 후처리입니다. pred_*는 정합 전 시점별 관측입니다.\n같은 축척으로 비교하세요. 선 개수 감소는 정확도 향상을 뜻하지 않습니다. 전체 도로망의 연결 안전성을 인증하지 않습니다.\n좌표 EPSG:32652, 높이 포함 XYZ입니다. 다섯 확장자 shp/shx/dbf/prj/cpg를 함께 보관하세요.\nCheckpoint·epoch·해시·프레임·confidence는 export-summary.json, 병합 규칙과 채택 이력은 별도 JSON에 있습니다.\n',encoding='utf-8')
 with (out/'README_KO.txt').open('a') as f:f.write('중심선은 차로 가운데의 주행 기준선, 좌우 경계는 그 차로 폭을 둘러싼 선입니다. ltype/rtype은 모델의 경계 유형 번호(0=표시 없음, 1=실선, 2=점선), -1은 참조용 미지정 값입니다. n_src는 병합에 사용한 관측 수입니다.\n새 분할은 train228/val14/test57입니다. 작은 검증 구간, 한 주행·한 환경, 파생 GT, 희소한 LiDAR 및 일부 미확인 유형이라는 한계가 있습니다. 이 SHP는 실제 운전에 사용할 검증된 HD 지도가 아닙니다.\n')
 with (out/'README_KO.txt').open('a') as f:f.write('좌우는 각 차로의 진행 방향 기준입니다. 일반 QGIS 2D 화면에서 높이가 보이지 않아도 원본 XYZ는 보존되어 있습니다. 정합은 여러 시점의 예측 대응·중복 병합이며 정답에 강제로 맞추는 작업이 아닙니다. 전후 위치오차와 GT 근접 길이 비교는 map-quality-comparison.json을 확인하세요.\n')
 with (out/'README_KO.txt').open('a') as f:f.write('gt_reference는 제공 NT2_LINK 원본을 test 관측 범위로 잘라낸 참조선입니다. 프레임마다 나눈 학습 GT 인스턴스와 선 개수·길이 정의가 다를 수 있습니다. 경계 유형0은 노면 표시가 없는 가상 경계 예측일 수 있습니다. Z 값은 보존했지만 수직 기준면을 별도 인증하지 않았습니다.\n')
 archive=run/'qgis-export-v1.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.iterdir():z.write(p,p.name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 summary['exports']={'summary':str((out/'export-summary.json').relative_to(STORE)),'zip':str(archive.relative_to(STORE))};summary.setdefault('previews',[]).append(str((out/'preview.png').relative_to(STORE)));write_json(run/'summary.json',summary)
 print('SHP_EXPORT_VERIFIED',archive,flush=True)
 finalize_if_ready()
if __name__=='__main__':main()
