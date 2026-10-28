"""Prepare actual SHP, calibrated RGB and timestamp-local LAS inputs on D:."""
from pathlib import Path
import json,subprocess,hashlib,collections,math,time
import numpy as np,cv2,shapefile,laspy
from .common import ROOT,STORE,RAW,SHP,PYTHON,status,write_json,read_json
cv2.setNumThreads(2)
def poses(path):
    rows=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        v=line.split(';');rows.append({'filename':v[0],'timestamp':float(v[1]),'position':np.array(v[2:5],float),'rotation':np.array(v[8:17],float).reshape(3,3)})
    return rows

def ensure_gt():
    status('gt',message='Building full Track B semantic/topology GT on D:')
    book=json.loads((ROOT.parent/'gg49-lanesegnet-gt-builder/configs/gg49-codebook.json').read_text())
    rd=shapefile.Reader(str(SHP/'RM1_LANELINE.shp'),encoding='cp949')
    codes=collections.Counter(str(x.as_dict()['LineType']) for x in rd.records())
    if not set(codes)<=set(book['line_type_codes']):raise RuntimeError('Unreviewed SHP LineType codes')
    links=shapefile.Reader(str(SHP/'NT2_LINK.shp'),encoding='cp949');directions=set(str(x.as_dict()['Direction']) for x in links.records())
    if not directions<=set(book['direction_codes']):raise RuntimeError('Unknown direction code')
    # Other(999) and mixed markings remain unknown without independent stroke evidence.
    book['line_type_codes'].update({'999':'unknown','223':'unknown','224':'unknown'})
    book['dataset_scope']='Local APR01 supplied map. Schema/codes checked; 999 and mixed codes masked.'
    book['local_input_sha256']={n:hashlib.sha256((SHP/n).read_bytes()).hexdigest() for n in ['RM1_LANELINE.shp','RM1_LANELINE.dbf','NT2_LINK.shp','NT2_LINK.dbf']}
    bookpath=ROOT/'configs/codebook.local.json';write_json(bookpath,book)
    work=STORE/'artifacts/gt-track-b-v1'
    complete=read_json(work/'GT_BUILD_RUN.json').get('status')=='complete'
    if not complete:
        if (work/'MANIFEST.json').exists():
            # Never overwrite partial GT; retain provenance and use a fresh run.
            work=STORE/'artifacts'/('gt-track-b-'+str(int(time.time())))
        subprocess.run([PYTHON,str(ROOT.parent/'gg49-lanesegnet-gt-builder/scripts/build_gt.py'),'--stage','all','--tracks','B','--shp-encoding','cp949','--shp-root',str(SHP),'--raw-root',str(RAW),'--work',str(work),'--codebook',str(bookpath)],check=True)
    caches=list(work.glob('semantic_topology_*'))
    assert len(caches)==1
    rows=read_json(work/'MANIFEST.json')['frames']
    cams=[poses(RAW/f'Track_B0{i}/External Orientation.csv') for i in range(1,7)]
    for row in rows:
        idx=row['frame'];row['gt_path']=str(caches[0]/(row['key']+'.npz'));row['camera_files']=[str(RAW/f'Track_B0{i+1}'/cam[idx]['filename']) for i,cam in enumerate(cams)]
        with np.load(row['gt_path'],allow_pickle=False) as d:
            n=len(d['lanes']);assert d['lanes'].shape==(n,3,10,3) and d['adjacency'].shape==(n,n)
            row['lanes']=n
        row['geometry_eligible']=bool(row['supervised_eligible'])
    write_json(STORE/'artifacts/prepared/FRAMES.json',{'gt_work':str(work),'frames':rows})
    write_json(STORE/'reports/codebook-local-audit.json',{'codes':dict(codes),'direction_codes':sorted(directions),'masked_codes':['223','224','999'],'input_sha256':book['local_input_sha256']})
    return rows,cams

def prepare_rgb(rows,cams):
    status('qa',message='Caching six undistorted RGB views and consistent camera projection matrices')
    dest=STORE/'cache/rgb';dest.mkdir(parents=True,exist_ok=True);cal=[]
    for i in range(1,7):
        text=(RAW/f'Track_B0{i}/Internal Orientation.txt').read_text()
        vals={k:float(v) for line in text.splitlines() if '=' in line for k,v in [line.split('=',1)] if k.startswith('Dist_CV_')}
        K=np.array([[vals['Dist_CV_FX'],0,vals['Dist_CV_CX']],[0,vals['Dist_CV_FY'],vals['Dist_CV_CY']],[0,0,1.]])
        D=np.array([vals['Dist_CV_K1'],vals['Dist_CV_K2'],vals['Dist_CV_P1'],vals['Dist_CV_P2'],vals['Dist_CV_K3'],vals['Dist_CV_K4'],vals['Dist_CV_K5'],vals['Dist_CV_K6']])
        newK=K.copy();newK[0]*=384/vals['Dist_CV_WIDTH'];newK[1]*=384/vals['Dist_CV_HEIGHT']
        maps=cv2.initUndistortRectifyMap(K,D,None,newK,(384,384),cv2.CV_32FC1);cal.append((newK,maps))
    for k,row in enumerate(rows):
        target=dest/(row['key']+'.npz')
        if target.exists():continue
        R=np.array(row['rotation']);origin=np.array(row['origin']);images=[];matrices=[]
        for i,(K,maps) in enumerate(cal):
            image=cv2.imread(row['camera_files'][i])
            if image is None:raise FileNotFoundError(row['camera_files'][i])
            image=cv2.remap(image,*maps,cv2.INTER_LINEAR)
            pose=cams[i][row['frame']];c2e=R.T@pose['rotation']@np.diag([1,-1,-1]);t=R.T@(pose['position']-origin)
            e2c=np.eye(4);e2c[:3,:3]=c2e.T;e2c[:3,3]=-c2e.T@t
            intrinsic=np.eye(4);intrinsic[:3,:3]=K
            matrices.append(intrinsic@e2c);images.append(image[:,:,::-1])
        np.savez(target,rgb=np.stack(images).astype(np.uint8),matrices=np.stack(matrices).astype(np.float32))
        if k%50==0:print('RGB cached',k,len(rows),flush=True)

def prepare_lidar(rows):
    dest=STORE/'cache/lidar';dest.mkdir(parents=True,exist_ok=True)
    reportpath=STORE/'reports/lidar-time-audit.json'
    if reportpath.exists() and all((dest/(r['key']+'.npz')).exists() for r in rows):return read_json(reportpath)['point_counts']
    status('qa',message='Rasterizing LAS points within +/-0.1s of each image timestamp (no accumulated-map fallback)')
    n=len(rows);times=np.array([r['timestamp'] for r in rows]);origins=np.array([r['origin'] for r in rows]);rotations=np.array([r['rotation'] for r in rows])
    count=np.zeros((n,20000),np.uint32);isum=np.zeros((n,20000),np.float64);low=np.full((n,20000),np.inf,np.float32);high=np.full_like(low,-np.inf);alltimes=[];files=[]
    for p in sorted((RAW/'Scanfiles').glob('Track_B_*.las')):
        with laspy.open(p) as reader:
            if reader.header.point_format.id!=3:raise RuntimeError('Unreviewed LAS format')
            encoding=int(reader.header.global_encoding.value)
            if encoding&1:raise RuntimeError('LAS adjusted GPS time differs from EO week-seconds; explicit conversion required')
            filemin=float('inf');filemax=float('-inf')
            for c in reader.chunk_iterator(1_000_000):
                ts=np.asarray(c.gps_time);assert np.isfinite(ts).all();filemin=min(filemin,float(ts.min()));filemax=max(filemax,float(ts.max()))
                right=np.searchsorted(times,ts).clip(0,n-1);left=(right-1).clip(0,n-1);ids=np.where(abs(times[left]-ts)<abs(times[right]-ts),left,right)
                valid=np.abs(times[ids]-ts)<=.100000001
                if not valid.any():continue
                ids=ids[valid];xyz=np.stack((np.asarray(c.x)[valid],np.asarray(c.y)[valid],np.asarray(c.z)[valid]),1);inten=np.asarray(c.intensity)[valid]
                for idx in np.unique(ids):
                    take=ids==idx;ego=(xyz[take]-origins[idx])@rotations[idx];iv=inten[take]
                    good=(ego[:,0]>=-51.2)&(ego[:,0]<51.2)&(ego[:,1]>=-25.6)&(ego[:,1]<25.6)&(ego[:,2]>=-4)&(ego[:,2]<=2)
                    ego=ego[good];iv=iv[good]
                    ix=np.floor((ego[:,0]+51.2)/.512).astype(int);iy=np.floor((ego[:,1]+25.6)/.512).astype(int);pix=iy*200+ix
                    np.add.at(count[idx],pix,1);np.add.at(isum[idx],pix,iv)
                    np.minimum.at(low[idx],pix,ego[:,2]);np.maximum.at(high[idx],pix,ego[:,2])
            files.append({'path':str(p),'gps_range':[filemin,filemax],'gps_global_encoding':encoding});print('LAS processed',p.name,flush=True)
    points=count.sum(1).tolist()
    if sum(v>0 for v in points)<n//2:raise RuntimeError('Too few frames have synchronized LiDAR; inspect time/coordinate alignment')
    for idx,row in enumerate(rows):
        observed=count[idx]>0;mean=np.divide(isum[idx],count[idx],out=np.zeros(20000),where=observed)
        bev=np.stack([np.log1p(count[idx]).clip(0,10)/10,(mean/65535).clip(0,1),np.where(observed,np.clip(low[idx]/6,-1,1),0),np.where(observed,np.clip(high[idx]/10,-1,1),0),observed.astype(np.float32)]).reshape(5,100,200).astype(np.float16)
        np.savez_compressed(dest/(row['key']+'.npz'),bev=bev,point_count=np.array(points[idx]))
    write_json(reportpath,{'image_time_range':[float(times.min()),float(times.max())],'half_window_seconds':.1,'files':files,'point_counts':points,'frames_with_points':sum(v>0 for v in points),'mode':'timestamp-local, not full-track accumulated map'})
    return points

def qa(rows):
    folder=STORE/'reports/input-qa';folder.mkdir(parents=True,exist_ok=True)
    eligible=[r for r in rows if r['joint_eligible']];chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(12,len(eligible))).astype(int)]
    for row in chosen:
        with np.load(STORE/'cache/rgb'/(row['key']+'.npz')) as d:rgb=d['rgb'];mats=d['matrices']
        with np.load(row['gt_path']) as d:lanes=d['lanes']
        pictures=[]
        for image,P in zip(rgb,mats):
            image=image[:,:,::-1].copy()
            for lane in lanes:
                for k,color in enumerate([(170,70,180),(0,150,240),(80,190,70)]):
                    a=np.c_[lane[k],np.ones(10)]@P.T
                    for j in range(9):
                        if min(a[j,2],a[j+1,2])>.1:
                            uv=np.rint(a[j:j+2,:2]/a[j:j+2,2:3]).clip(-10000,10000).astype(int)
                            cv2.line(image,tuple(uv[0]),tuple(uv[1]),color,2,cv2.LINE_AA)
            pictures.append(image)
        montage=np.vstack([np.hstack(pictures[:3]),np.hstack(pictures[3:])]);cv2.putText(montage,row['key']+' | SHP GT projection (not prediction)',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        cv2.imwrite(str(folder/(row['key']+'.jpg')),montage)
    return ['reports/input-qa/'+r['key']+'.jpg' for r in chosen]

def main():
    status('data',message='Reading original D: inputs; all generated outputs are on D:')
    done=STORE/'artifacts/prepared/DATASET.json'
    if done.exists():print('Prepared dataset exists, reuse validated cache',flush=True);return
    rows,cams=ensure_gt();prepare_rgb(rows,cams);points=prepare_lidar(rows)
    xs=np.array([r['origin'][0] for r in rows]);a,b=np.percentile(xs,[60,80])
    for i,row in enumerate(rows):
        x=xs[i];row['lidar_points']=points[i];row['joint_eligible']=row['geometry_eligible'] and points[i]>0
        row['split']='unlabelled' if not row['joint_eligible'] else ('train' if x<a-75 else 'val' if a+75<x<b-75 else 'test' if x>b+75 else 'buffer')
    from .split_policy import apply_local_split
    split_audit=apply_local_split(rows)
    counts=split_audit['split_counts']
    if min(counts.get(k,0) for k in ['train','val','test'])<8:raise RuntimeError('Insufficient spatially independent split')
    # Bounding circles of the BEV ROI have radius57.25m. 150m X gap is larger than two radii.
    for one,two in [('train','val'),('val','test'),('train','test')]:
        x1=xs[[r['split']==one for r in rows]];x2=xs[[r['split']==two for r in rows]]
        assert np.min(np.abs(x1[:,None]-x2[None,:]))>114.5
    previews=qa(rows)
    write_json(done,{'frames':rows,'split_counts':counts,'split_audit':split_audit,'spatial_x_boundaries':[float(a),float(b)],'buffer_per_side_m':75,'camera_order':['01','02','03','04','05','06'],'pc_range':[-51.2,-25.6,-4,51.2,25.6,2],'bev_shape':[100,200],'previews':previews,'comparison_frame_policy':'same synchronized-LiDAR-available supervised frames for camera and fusion'})
    write_json(STORE/'reports/input-validation.json',{'status':'complete','split_counts':counts,'frames':len(rows),'previews':previews,'notes':['Image overlays require visual QA before claiming calibration accuracy.','No official OpenLane-V2 scores are implied by the custom GG49 split.']})
    print('DATA_PREPARATION_COMPLETE',counts,flush=True)
if __name__=='__main__':main()
