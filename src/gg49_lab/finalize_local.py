"""Final report only after real checkpoints, metrics, evaluation and SHP checks exist."""
import hashlib,json,zipfile,subprocess,datetime
import fiona
import torch
from .common import ROOT,STORE,STORAGE,read_json,write_json,now
def main():
 rows=[]
 for mode in ['camera','fusion']:
  name=STORAGE.get('run_prefix','')+mode+'-24e';run=STORE/'runs'/name
  summary=read_json(run/'summary.json');assert summary['status']=='completed' and summary['epoch']==24
  metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
  assert len({x['epoch'] for x in metrics if x['kind']=='epoch'})==24
  steps=[x['step'] for x in metrics if x['kind']=='step']
  assert set(steps)==set(range(1,summary['step']+1))
  ck=torch.load(run/'last.pth',map_location='cpu');assert ck['epoch']==24
  assert all(k in ck for k in ['model','optimizer','scaler','scheduler','rng_python','rng_numpy','rng_torch','rng_cuda','order','position','settings'])
  assert ck['settings']['dataset_sha256']==summary['settings']['dataset_sha256'];del ck
  export=read_json(run/'qgis-export-v1/export-summary.json');assert export['status']=='completed'
  for layer in export['layers']:
   with fiona.open(run/'qgis-export-v1'/(layer['layer']+'.shp')) as shp:
    assert len(shp)==layer['count'] and shp.crs.to_epsg()==32652 and shp.schema['geometry']=='3D LineString'
    assert set(shp.schema['properties'])=={'id','frame','score','ltype','rtype','n_src'}
    assert all(len(point)==3 for feature in shp for point in feature['geometry']['coordinates'])
  with zipfile.ZipFile(run/'qgis-export-v1.zip') as z:assert z.testzip() is None
  elapsed=(datetime.datetime.fromisoformat(summary['completed_at'])-datetime.datetime.fromisoformat(summary['started_at'])).total_seconds()
  rows.append({'run':name,'summary':summary,'evaluation':read_json(run/'evaluation.json'),'export':export,'optimizer_steps_in_metrics':len(set(steps)),'replayed_metric_rows_preserved':len(steps)-len(set(steps)),'wall_elapsed_seconds':elapsed,'map_quality':read_json(run/'qgis-export-v1/map-quality-comparison.json')})
 report={'status':'verified_complete','verified_at':now(),'runs':rows,'resume_test':read_json(STORE/'reports/resume-test.json'),'input_qa':read_json(STORE/'reports/QA_APPROVED.json'),'not_official_OpenLaneV2_score':True}
 lines=['# 이 PC의 새 Track B 학습 결과','', '실제 실행 완료 시점: '+now(),'','기존 작성자의 결과를 재사용하지 않은 '+STORAGE.get('run_prefix','')+' 실험이다. 로컬 F1이며 공식 OpenLane-V2 점수가 아니다.','']
 lines[0]='# 이 PC의 새 Track B 학습 결과'
 for row in rows:
  s=row['summary'];e=row['evaluation']['results'];x=row['export']
  lines += ['## '+row['run'],f"- 학습: 24 epoch / {s['step']} optimizer updates",f"- Best: epoch {s['best_epoch']}, validation loss {s['best_val']}",f"- Test F1@1m: {e['test']['geometry']['1.0m']['f1']}",f"- Test F1@0.5m: {e['test']['geometry']['0.5m']['f1']}",f"- 원시 예측 {x['raw_count']}개 / 병합 {x['merged_count']}개. 병합 정확도 향상을 뜻하지 않는다.",'- SHP: runs/'+row['run']+'/qgis-export-v1/','- ZIP: runs/'+row['run']+'/qgis-export-v1.zip','']
 lines+=['두 모델 차이를 실제 test 지표로 비교해야 하며, loss 하락이나 학습 완료만으로 LiDAR 개선을 주장하지 않는다.','작은 Track B 공간 분할과 파생 GT라는 한계가 있다. 병합은 GT 없는 후처리이며 전체 도로망 연결 보증이 아니다.']
 text='\n'.join(lines)+'\n';(STORE/'reports/LOCAL_TRAINING_RESULTS.md').write_text(text)
 exports=STORE/'exports';exports.mkdir(exist_ok=True)
 shared=exports/(STORAGE.get('run_prefix','')+'qgis-comparison.zip')
 with zipfile.ZipFile(shared,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('README_KO.txt','이번 PC에서 새로 학습한 24 epoch 비교 실험입니다. camera/는 영상 전용, fusion/은 RGB+LiDAR입니다. 각 폴더의 pred_centers.shp, pred_left.shp, pred_right.shp는 예측이며 gt_reference.shp는 정답 참조선입니다. merged_*는 GT 없이 병합한 별도 후처리입니다.\n모든 레이어 EPSG:32652/XYZ. 각 SHP의 shp/shx/dbf/prj/cpg를 함께 보관하세요. QGIS에서 같은 모델 폴더의 예측과 정답을 함께 열고 우클릭-레이어로 확대하여 비교하세요. checkpoint 해시, epoch, test 시점, confidence0.5는 폴더별 export-summary.json을 보세요.\n실제 평가와 한계는 LOCAL_TRAINING_RESULTS.md에 있습니다. 선 감소는 정확도 개선을 의미하지 않습니다.\n')
  z.writestr('LOCAL_TRAINING_RESULTS.md',text)
  for row in rows:
   source=STORE/'runs'/row['run']/'qgis-export-v1'
   for p in sorted(source.rglob('*')):
    if p.is_file():z.write(p,row['summary']['modality']+'/'+str(p.relative_to(source)))
 with zipfile.ZipFile(shared) as z:assert z.testzip() is None
 report['shared_zip']={'path':str(shared),'sha256':hashlib.sha256(shared.read_bytes()).hexdigest(),'bytes':shared.stat().st_size}
 with (ROOT/'docs/mms_lanesegnet_training_experiment.md').open('a') as f:f.write('\n\n## 새 PC 실행 최종 검증\n\n'+text)
 with (ROOT/'docs/LOCAL_WINDOWS_WSL_gs.md').open('a') as f:f.write('\n\n'+text)
 guide=ROOT/'docs/LOCAL_WINDOWS_WSL_gs.md'
 guide_text=guide.read_text().replace('환경·데이터 준비와 초기 1 epoch 시연 및 SHP 검증을 완료했다. 라벨 분포 점검 후 v2 분할로 시연 및 24 epoch 비교 학습을 새로 진행한다. 최종 완료 여부와 수치는 SSD의 runs/reports가 근거다.','영상/결합 모델 각24epoch, best checkpoint test 평가와 SHP 재읽기 검증을 완료했다. 최종 전체 검증의 근거는 reports/LOCAL_COMPLETION_AUDIT.json이다. 아래 과정 기록과 최종 결과를 구분해 확인한다.')
 guide.write_text(guide_text);(STORE/'RUN_GUIDE_KO.md').write_text(guide_text)
 # Capture the actual local implementation, including new untracked scripts.
 base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).split(b'\0')
 snapshot=STORE/'reports/LOCAL_SOURCE_SNAPSHOT.zip';manifest={}
 with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
  for raw in sorted(set(files)):
   if not raw:continue
   relative=raw.decode();p=ROOT/relative
   if not p.is_file() or p.suffix in ['.pth','.las','.npz','.zip']:continue
   assert p.stat().st_size<10*1024*1024, 'Source snapshot unexpectedly contains a large file'
   z.write(p,relative);manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
 with zipfile.ZipFile(snapshot) as z:assert z.testzip() is None
 with (STORE/'reports/LOCAL_TRACKED_CHANGES.patch').open('wb') as f:subprocess.run(['git','diff','--binary'],cwd=ROOT,stdout=f,check=True)
 write_json(STORE/'reports/LOCAL_SOURCE_SNAPSHOT.json',{'base_commit':base,'snapshot':str(snapshot),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'files':manifest,'note':'Source and local configuration only; no original data, weights, environment or old experiment output.'})
 write_json(STORE/'reports/LOCAL_COMPLETION_AUDIT.json',report)
 print('LOCAL_COMPLETION_AUDIT_VERIFIED',flush=True)
if __name__=='__main__':main()
