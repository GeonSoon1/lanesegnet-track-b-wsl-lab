"""Copy Track B training inputs only; never modify the original drive."""
from pathlib import Path
import hashlib,json,shutil,time
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('D:/2026-APR-01_GG803SEC012');SHP=Path('D:/SHP데이터/통합SHP')
DEST=Path(json.loads((ROOT/'configs/storage.local.json').read_text(encoding='utf-8'))['storage_root_windows'])/'data/track-b-full';DEST.mkdir(parents=True,exist_ok=True)
manifest={'source_raw':str(RAW),'source_shp':str(SHP),'files':[],'status':'running'}
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(src.read_bytes()).hexdigest() if src.stat().st_size<10_000_000 else None
    if not dst.exists() or dst.stat().st_size!=src.stat().st_size:shutil.copyfile(src,dst)
    def hashfile(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    digest=digest or hashfile(src)
    if hashfile(dst)!=digest:raise RuntimeError('Copy hash mismatch: '+str(src))
    manifest['files'].append({'source':str(src),'relative':str(dst.relative_to(DEST)),'bytes':src.stat().st_size,'sha256':digest})
for stem in ['RM1_LANELINE','NT2_LINK']:
    for ext in ['.shp','.shx','.dbf','.prj','.cpg']:copy(SHP/(stem+ext),DEST/'shp'/(stem+ext))
for sensor in ['Sphere','01','02','03','04','05','06']:
    source=RAW/f'Track_B{sensor}';target=DEST/'raw'/source.name
    copy(source/'External Orientation.csv',target/'External Orientation.csv')
    if (source/'Internal Orientation.txt').exists():copy(source/'Internal Orientation.txt',target/'Internal Orientation.txt')
    if sensor!='Sphere':
        rows=[line.split(';') for line in (source/'External Orientation.csv').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        for i,row in enumerate(rows):
            copy(source/row[0],target/row[0])
        print(json.dumps({'sensor':sensor,'images':len(rows),'status':'copied'}),flush=True)
for src in sorted((RAW/'Scanfiles').glob('Track_B_*.las')):
    copy(src,DEST/'raw/Scanfiles'/src.name)
    print('Copied LAS '+src.name,flush=True)
manifest['status']='complete';manifest['bytes']=sum(f['bytes'] for f in manifest['files'])
(DEST/'STAGING_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'complete','files':len(manifest['files']),'bytes':manifest['bytes']}),flush=True)