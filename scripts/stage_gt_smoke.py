"""Windows: copy only SHP + uniformly sampled EO for a geometry smoke run."""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='track-b-smoke12')
    ap.add_argument('--count',type=int,default=12)
    ap.add_argument('--raw-root',type=Path,default=Path('D:/2026-APR-01_GG803SEC012'))
    ap.add_argument('--shp-root',type=Path,default=Path('D:/SHP데이터/통합SHP'))
    args=ap.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():
        ap.error('Use a simple run name')
    root=Path(__file__).resolve().parents[1]
    storage=json.loads((root/'configs/storage.local.json').read_text(encoding='utf-8'))
    dest=Path(storage['storage_root_windows'])/'data'/args.run
    if dest.exists():
        raise FileExistsError('Choose a new --run; staged inputs already exist')
    streams=['Sphere','01','02','03','04','05','06']
    sources={s:args.raw_root/f'Track_B{s}'/'External Orientation.csv' for s in streams}
    rows={s:[line for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()] for s,p in sources.items()}
    total=len(rows['Sphere'])
    if not 2<=args.count<=total or any(len(r)!=total for r in rows.values()):
        raise ValueError('Invalid sample count or EO row mismatch')
    indices=[round(i*(total-1)/(args.count-1)) for i in range(args.count)]
    shpfiles=[args.shp_root/(stem+ext) for stem in ['RM1_LANELINE','NT2_LINK'] for ext in ['.shp','.shx','.dbf','.prj','.cpg']]
    if not all(p.is_file() for p in shpfiles):
        raise FileNotFoundError('Required SHP sidecar missing')
    (dest/'shp').mkdir(parents=True)
    hashes=[]
    for src in shpfiles:
        target=dest/'shp'/src.name
        shutil.copyfile(src,target)
        digest=sha(src)
        assert sha(target)==digest
        hashes.append({'path':str(src),'sha256':digest,'bytes':src.stat().st_size})
    for stream,src in sources.items():
        folder=dest/'raw'/f'Track_B{stream}'
        folder.mkdir(parents=True)
        (folder/'External Orientation.csv').write_text('\n'.join(rows[stream][i] for i in indices)+'\n',encoding='utf-8')
        hashes.append({'path':str(src),'sha256':sha(src),'bytes':src.stat().st_size})
    manifest={'purpose':'Geometry smoke only: sampled EO, no RGB/LAS copied; not a training dataset','raw_root':str(args.raw_root),'shp_root':str(args.shp_root),'track':'B','source_frame_count':total,'sampled_frames':args.count,'indices':[{'staged_index':k,'original_index_zero_based':i,'sphere_filename':rows['Sphere'][i].split(';')[0]} for k,i in enumerate(indices)],'input_hashes':hashes}
    (dest/'stage-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(dest))

if __name__=='__main__':
    main()
