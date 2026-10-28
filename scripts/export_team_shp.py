"""Export existing verified follow-up GIS results to a portable SHP team bundle. No inference/training."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import fiona
import numpy as np
from fiona.crs import CRS
ROOT=Path(__file__).resolve().parents[1]
STORE=Path('/mnt/d/mms_experiments/gg49-lanesegnet-lab_gs')
SRC=STORE/'reports/followup-45-v1'
OUT=STORE/'exports/team-share-v1'

def main():
    if OUT.exists():
        raise RuntimeError('Export exists; preserve it and choose a new version for another export')
    OUT.mkdir(parents=True)
    checks=[]
    for gpkg in sorted((SRC/'maps').glob('*/map.gpkg')):
        folder=OUT/'shp'/gpkg.parent.name
        folder.mkdir(parents=True)
        for layer in fiona.listlayers(gpkg):
            with fiona.open(gpkg,layer=layer) as source:
                schema={'geometry':source.schema['geometry'],'properties':{
                    k:('str:254' if v.startswith('str') else 'float:24.15' if v.startswith('float') else 'int:18')
                    for k,v in source.schema['properties'].items()}}
                assert all(len(k)<=10 for k in schema['properties'])
                features=list(source)
                target=folder/(layer+'.shp')
                with fiona.open(target,'w',driver='ESRI Shapefile',schema=schema,crs_wkt=source.crs_wkt,encoding='UTF-8') as dest:
                    for feature in features:
                        props=dict(feature['properties'])
                        assert all(not isinstance(v,str) or len(v.encode('utf-8'))<=254 for v in props.values())
                        dest.write({'geometry':dict(feature['geometry']),'properties':props})
            with fiona.open(target,encoding='UTF-8') as actual:
                assert actual.crs==CRS.from_epsg(32652)
                assert actual.schema['geometry']=='3D LineString'
                after=list(actual)
                assert len(after)==len(features)
                max_delta=0.
                for before,a in zip(features,after):
                    bxyz=np.asarray(before['geometry']['coordinates']);axyz=np.asarray(a['geometry']['coordinates'])
                    assert bxyz.shape==axyz.shape and bxyz.shape[1]==3
                    delta=float(np.abs(bxyz-axyz).max());assert delta<=1e-9
                    max_delta=max(max_delta,delta)
                    for k,v in before['properties'].items():
                        if isinstance(v,float):assert abs(a['properties'][k]-v)<=1e-10
                        else:assert a['properties'][k]==v
            checks.append({'path':str(target.relative_to(OUT)),'features':len(features),'crs':'EPSG:32652','geometry':'3D LineString','max_xyz_delta_m':max_delta,'attributes_verified':True})
        for name in ['association-ledger.json','metrics.json','before-after.png','spread.png','observations.json','merged.json','map.gpkg']:
            p=gpkg.parent/name
            if p.exists():shutil.copy2(p,folder/name)
    for name in ['CONFIG_SUMMARY.md','TEAM_BRIEF.md','mms_lanesegnet_training_experiment.md']:
        shutil.copy2(ROOT/'docs'/name,OUT/name)
    for name in ['protocol.json','verification.json','EXPERIMENTS_4_5.md']:
        if (SRC/name).exists():shutil.copy2(SRC/name,OUT/name)
    readme="""# GG49 팀 공유 — SHP 결과 열기

이 ZIP을 모두 압축 해제한 뒤 QGIS에 .shp 파일을 드래그하세요. .shp만 따로 전달하지 마세요. 같은 이름의 .shx/.dbf/.prj/.cpg가 함께 필요합니다.

## 우선 볼 파일
1. CONFIG_SUMMARY.md: 실제 학습 조건과 코드 출처
2. TEAM_BRIEF.md: 발표용 쉬운 설명
3. shp/A_fusion/before-after.png: RGB+LiDAR 정합 전후 그림
4. shp/A_fusion/observations.shp: 시점별 예측 중앙선(정합 전)
5. shp/A_fusion/merged_centers.shp: 정합 후 중앙선
6. shp/A_fusion/merged_left.shp, merged_right.shp: 예측된 좌우 경계
7. shp/A_fusion/gt_reference.shp: 평가에 사용한 원본 SHP 참조선. 모델 예측이 아닙니다.

## 폴더 뜻과 한계
- camera: 영상 전용; fusion: 영상+LiDAR.
- A: Track B의 B_0082~B_0101, B: Track B의 B_0102~B_0121. 각 20시점 validation 구간입니다. Track A 결과가 아닙니다.
- *_supported: 두 번 이상 관측한 부분을 확정한 추가 변형. uncertain_centers.shp는 확정하지 않은 부분입니다.
- 기본 평균은 A 두 모델만 사전 기준 통과(2/4); B 두 모델은 위치오차 증가로 실패. supported는 누락 증가로 모두 실패. 성공한 완성 지도라고 배포하면 안 됩니다.
- 두 모델은 실제 각각 24 epoch 학습했고, 첫 held-out test F1@1m은 둘 다 0.284672였습니다. LiDAR가 더 좋다는 증거를 얻지는 못했습니다.
- 이 SHP는 후속 실험의 학습 경로와 맞춘 추론 reference 결과를 변환한 것입니다. 원래 test 결과 또는 전체 Track 결과가 아닙니다. 변환 시 재학습/재추론하지 않았습니다.

## QGIS에서 비교하는 방법
좌표계는 EPSG:32652 (WGS84 / UTM zone 52N), XY 단위 m, Z 좌표도 보존한 3D LineString입니다. Z의 수직 기준은 원 실험 값을 그대로 따르며 독립적으로 검증한 고도 datum을 새로 부여하지 않았습니다.
QGIS 프로젝트 좌표계를 EPSG:32652로 선택하고 레이어 우클릭 → 레이어로 확대하세요. observations는 연한 회색, merged_centers는 진한 파랑, left/right는 주황/초록, gt_reference는 보라로 지정하면 비교하기 쉽습니다. 정합 전후는 레이어 체크를 켜고 끄며 비교하세요. 서로 다른 모델/구간/변형은 따로 켜세요.
QGIS 없이도 각 폴더 before-after.png를 볼 수 있습니다. 지도 서버나 대시보드를 실행할 필요가 없습니다.

## 속성
id: 관측 또는 병합선 ID; score: 예측/결합 confidence; sources: 결합에 참여한 관측 수(모든 vertex의 지지 횟수라는 뜻이 아님); left_type/right_type: 0 none, 1 solid, 2 dashed, -1 unknown. gt_reference에는 원본 ID만 있습니다.
SHP에는 중앙선 관측을 넣었으며 merged_left/right는 모델의 예측 경계입니다. 공식 HD 지도 topology를 검증한 산출물은 아닙니다. 상세 대응/관측 추적은 각 association-ledger.json 및 observations.json/merged.json에 보존했습니다. 원본 다층 파일은 map.gpkg로 함께 제공합니다.

export-verification.json에서 각 SHP의 건수·CRS·XYZ·속성 비교 검증을 확인할 수 있습니다. manifest.json은 배포 파일 SHA256입니다.
"""
    (OUT/'README_KO.md').write_text(readme,encoding='utf-8')
    report={'status':'passed','source':str(SRC),'layer_count':len(checks),'feature_count':sum(x['features'] for x in checks),'checks':checks}
    (OUT/'export-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUT.parent)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'layers':len(checks),'features':report['feature_count'],'zip_check':'passed'},indent=2))
if __name__=='__main__':main()
