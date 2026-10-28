"""Track B validation-only failure, LiDAR ablation and score-threshold diagnostics."""
import argparse,copy,csv,hashlib,json,logging,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from .common import STORE,ROOT,now,read_json,write_json,ensure_storage
from .diagnostic_metrics import THRESHOLDS,derangement,measure,combine,direction_free_geometry
OUT=STORE/'reports/diagnostics-track-b-v1'
STATE=STORE/'runs/diagnostics-status.json'
CONDITIONS=['camera','fusion_normal','fusion_zero','fusion_shuffled']
COLORS={'gt':'#25865a','camera':'#2879c8','fusion_normal':'#d58224','fusion_zero':'#8a58a8','fusion_shuffled':'#bd4a48'}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def state(phase,done=0,total=1,**extra):
    write_json(STATE,dict(status='running',phase=phase,completed=done,total=total,updated_at=now(),pid=os.getpid(),output=str(OUT.relative_to(STORE)),**extra))
    print(json.dumps({'phase':phase,'completed':done,'total':total},ensure_ascii=False),flush=True)

def read_pred(condition,key):
    with np.load(OUT/'predictions'/condition/(key+'.npz'),allow_pickle=False) as p:return {k:p[k].copy() for k in p.files}

def load_gt(row):
    with np.load(row['gt_path'],allow_pickle=False) as p:return p['lanes'].copy()

def gpu_check():
    n=int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if n>10752:raise RuntimeError(f'GPU budget exceeded: {n} MiB')
    return n

def infer(dataset,ids,donors):
    import torch
    from torch.cuda.amp import autocast
    from .model import LabModel,seed_all
    torch.set_num_threads(4)
    total=len(ids)*4;done=0;peak=read_json(OUT/'summary.json').get('gpu_peak_observed_mib',0)
    for mode,conditions in [('camera',['camera']),('fusion',CONDITIONS[1:])]:
        pending=any(not (OUT/'predictions'/c/(dataset.frames[i]['key']+'.npz')).exists() for c in conditions for i in ids)
        if not pending:done+=len(ids)*len(conditions);continue
        seed_all();logging.disable(logging.INFO)
        try:model=LabModel(mode,load_initial=False).cuda()
        finally:logging.disable(logging.NOTSET)
        run=STORE/'runs'/('camera-24e' if mode=='camera' else 'fusion-24e')
        checkpoint=torch.load(run/'best.pth',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==6
        model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;model.eval()
        captured={}
        def head_hook(module,args,output):
            captured['raw_lanes']=output['all_lanes_preds'][-1,0].detach().cpu().numpy().reshape(-1,3,10,3)
            captured['raw_scores']=output['all_cls_scores'][-1,0].sigmoid().detach().cpu().numpy().reshape(-1)
        def fusion_hook(module,args,output):
            before=args[0].float();res=output.float()-before;mask=args[1][:,4].flatten(1)>0
            obs=res[mask];base=before[mask]
            captured['fusion_stats']=np.array([float(res.square().mean().sqrt()),float(before.square().mean().sqrt()),float(obs.square().mean().sqrt()) if obs.numel() else 0.,float(base.square().mean().sqrt()) if base.numel() else 0.,int(mask.sum())],np.float64)
        hook=model.net.pts_bbox_head.register_forward_hook(head_hook)
        fh=model.fusion.register_forward_hook(fusion_hook) if mode=='fusion' else None
        for idx in ids:
            row=dataset.frames[idx];img,metas,target,lidar=dataset.get(idx)
            for condition in conditions:
                dest=OUT/'predictions'/condition/(row['key']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():done+=1;continue
                supplied=lidar
                if condition=='fusion_zero':supplied=torch.zeros_like(lidar)
                elif condition=='fusion_shuffled':
                    donor=dataset.frames[donors[idx]]
                    with np.load(STORE/'cache/lidar'/(donor['key']+'.npz'),allow_pickle=False) as d:supplied=torch.from_numpy(d['bev'].astype(np.float32)).unsqueeze(0).cuda()
                captured.clear()
                with torch.no_grad(),autocast():prediction,_=model.predict(img.clone(),copy.deepcopy(metas),supplied if mode=='fusion' else None)
                p=prediction[0]
                result=dict(lanes=np.asarray(p[0],np.float32).reshape(-1,3,10,3),scores=np.asarray(p[1],np.float32),**captured)
                assert all(np.isfinite(v).all() for v in result.values())
                # Repeated inference establishes the numerical baseline for input sensitivity.
                if condition=='fusion_normal' and idx in ids[:3]:
                    first={k:v.copy() for k,v in result.items()}
                    with torch.no_grad(),autocast():model.predict(img.clone(),copy.deepcopy(metas),lidar)
                    result['repeat_max_xyz_delta_m']=np.array(np.max(np.abs(captured['raw_lanes']-first['raw_lanes'])))
                    result['repeat_max_score_delta']=np.array(np.max(np.abs(captured['raw_scores']-first['raw_scores'])))
                    result['raw_lanes']=first['raw_lanes'];result['raw_scores']=first['raw_scores'];result['fusion_stats']=first['fusion_stats']
                temporary=dest.with_suffix('.tmp.npz');np.savez_compressed(temporary,**result);os.replace(temporary,dest)
                done+=1;peak=max(peak,gpu_check());state('inference',done,total,condition=condition,frame=row['key'],gpu_peak_observed_mib=peak)
            del img,metas,target,lidar
        hook.remove()
        if fh:fh.remove()
        del model;torch.cuda.empty_cache()
    return peak

def summarize(rows,donors,dataset):
    allcurves={};cases=[];sensitivity={};tolerances=[.5,1.,2.]
    for condition in CONDITIONS:
        curves=[]
        for confidence in THRESHOLDS:
            measurements={str(t)+'m':[] for t in tolerances}
            for row in rows:
                p=read_pred(condition,row['key']);gt=load_gt(row)
                for t in tolerances:measurements[str(t)+'m'].append(measure(p['lanes'],p['scores'],gt,confidence,t))
            curves.append({'confidence':confidence,'geometry':{k:combine(v) for k,v in measurements.items()}})
        allcurves[condition]=curves
    # Reproduce the existing baseline numbers without changing the previous evaluations.
    for condition,run in [('camera','camera-24e'),('fusion_normal','fusion-24e')]:
        expected=read_json(STORE/'runs'/run/'evaluation.json')['results']['val']['geometry']
        actual=next(r for r in allcurves[condition] if r['confidence']==.5)['geometry']
        for t in ['0.5m','1.0m']:
            for k in ['tp','predicted','gt']:assert actual[t][k]==expected[t][k],(condition,t,k,actual[t],expected[t])
    for row in rows:
        gt=load_gt(row);preds={c:read_pred(c,row['key']) for c in CONDITIONS}
        metrics={c:measure(p['lanes'],p['scores'],gt,.5,1.) for c,p in preds.items()}
        proximity={c:direction_free_geometry(p['lanes'][p['scores']>=.5],gt) for c,p in preds.items()}
        deltas={}
        normal=preds['fusion_normal']
        for c in ['fusion_zero','fusion_shuffled']:
            p=preds[c];delta=np.linalg.norm(normal['raw_lanes'][:,0]-p['raw_lanes'][:,0],axis=-1).mean(-1)
            score=np.abs(normal['raw_scores']-p['raw_scores']);high=normal['raw_scores']>=.5
            deltas[c]={'mean_query_center_delta_m':float(delta.mean()),'max_query_center_delta_m':float(delta.max()),'mean_query_score_delta':float(score.mean()),'max_query_score_delta':float(score.max()),'confident_query_mean_delta_m':float(delta[high].mean()) if high.any() else None,'changed_selected_query_count':int(np.sum((normal['raw_scores']>=.5)!=(p['raw_scores']>=.5)))}
        with np.load(STORE/'cache/lidar'/(row['key']+'.npz'),allow_pickle=False) as d:
            observed=int((d['bev'][4]>0).sum());points=int(d['point_count'])
        cases.append({'key':row['key'],'frame_index':row['frame'],'timestamp':row['timestamp'],'split':'val','metrics':metrics,'geometry_proximity':proximity,'gt_segment_lengths_m':np.linalg.norm(np.diff(gt[:,0],axis=1),axis=-1).sum(-1).tolist(),'prediction_segment_lengths_m':{c:np.linalg.norm(np.diff(p['lanes'][p['scores']>=.5,0],axis=1),axis=-1).sum(-1).tolist() for c,p in preds.items()},'input':{'lidar_points':points,'observed_cells':observed,'cells':20000,'coverage':observed/20000,'gt_nearest_m':row['GT_nearest_m']},'deltas':deltas,'repeat_max_xyz_delta_m':float(normal.get('repeat_max_xyz_delta_m',0)) if row in rows[:3] else None,'fusion_stats':{c:preds[c]['fusion_stats'].tolist() for c in CONDITIONS[1:]},'donor_key':dataset.frames[donors[next(i for i,r in enumerate(dataset.frames) if r['key']==row['key'])]]['key']})
    for c in ['fusion_zero','fusion_shuffled']:
        sensitivity[c]={k:float(np.mean([r['deltas'][c][k] for r in cases])) for k in ['mean_query_center_delta_m','max_query_center_delta_m','mean_query_score_delta','max_query_score_delta','changed_selected_query_count']}
    geometry={}
    for c in CONDITIONS:
        items=[r['geometry_proximity'][c] for r in cases]
        gl=sum(r['gt_length_m'] for r in items);pl=sum(r['pred_length_m'] for r in items)
        geometry[c]={'gt_supported_fraction':sum(r['gt_supported_length_m'] for r in items)/gl if gl else None,'pred_supported_fraction':sum(r['pred_supported_length_m'] for r in items)/pl if pl else None,'scope':'Direction-free, length-weighted proximity within1m. No1:1 association, no topology, not instance F1.'}
    best={c:max(v,key=lambda r:r['geometry']['1.0m']['f1']) for c,v in allcurves.items()}
    selected=[]
    for name in ['camera','fusion_normal']:
        limit=4 if name=='camera' else 8
        for case in sorted(cases,key=lambda r:(r['metrics'][name]['fn'],r['metrics'][name]['fp']),reverse=True):
            if case['key'] not in selected:selected.append(case['key'])
            if len(selected)>=limit:break
    for case in sorted(cases,key=lambda r:r['deltas']['fusion_zero']['mean_query_center_delta_m'],reverse=True):
        if case['key'] not in selected:selected.append(case['key'])
        if len(selected)>=10:break
    return {'geometry_proximity':geometry,'curves':allcurves,'sensitivity':sensitivity,'best_validation_thresholds':best,'cases':cases,'recommended_cases':selected,'reproduced_prior_validation_counts':True}

def project_segments(lanes,P):
    import cv2
    lines=[]
    for lane in lanes:
        for part in range(3):
            a=np.c_[lane[part],np.ones(10)]@P.T
            for j in range(9):
                if min(a[j,2],a[j+1,2])>.1:
                    uv=np.rint(a[j:j+2,:2]/a[j:j+2,2:3]).clip(-10000,10000).astype(int)
                    ok,p1,p2=cv2.clipLine((0,0,384,384),tuple(uv[0]),tuple(uv[1]))
                    if ok:lines.append([part,*map(int,p1),*map(int,p2)])
    return lines

def draw_lanes(ax,lanes,color,label):
    for n,lane in enumerate(lanes):
        for k in range(3):ax.plot(lane[k,:,0],lane[k,:,1],color=color,lw=1.8 if k==0 else .65,alpha=1 if k==0 else .55,label=label if n==0 and k==0 else None)
        if label=='SHP GT':
            ax.scatter(lane[0,[0,-1],0],lane[0,[0,-1],1],s=18,facecolors='white',edgecolors=color,zorder=5,linewidth=.8)
            middle=lane[0,5];ax.annotate('G'+str(n),middle[:2],xytext=(0,8),textcoords='offset points',color=color,fontsize=8)

def render(rows,analysis):
    import cv2,matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#f3f6f8','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    for index,row in enumerate(rows):
        key=row['key'];folder=OUT/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
        gt=load_gt(row);preds={c:read_pred(c,key) for c in CONDITIONS};selected={c:p['lanes'][p['scores']>=.5] for c,p in preds.items()}
        with np.load(STORE/'cache/rgb'/(key+'.npz'),allow_pickle=False) as d:rgb=d['rgb'].copy();mats=d['matrices'].copy()
        with np.load(STORE/'cache/lidar'/(key+'.npz'),allow_pickle=False) as d:bev=d['bev'].astype(np.float32)
        iy,ix=np.nonzero(bev[4]>0);xyz=np.column_stack([-51.2+(ix+.5)*.512,-25.6+(iy+.5)*.512,bev[2,iy,ix]*6])
        info=next(r for r in analysis['cases'] if r['key']==key)
        # Equal axes in all panels; cell centers are a diagnostic approximation, not raw LAS points.
        for filename,conditions in [('comparison.png',['camera','fusion_normal']),('ablation.png',CONDITIONS[1:])]:
            fig,axes=plt.subplots(1,len(conditions),figsize=(7*len(conditions),5.2),squeeze=False)
            for ax,c in zip(axes[0],conditions):
                ax.scatter(xyz[:,0],xyz[:,1],s=2,color='#8596a3',alpha=.3,label='Observed LiDAR cells')
                draw_lanes(ax,gt,COLORS['gt'],'SHP GT');draw_lanes(ax,selected[c],COLORS[c],c)
                m=info['metrics'][c];ax.set_title(f'{c} | TP {m["tp"]}, FP {m["fp"]}, FN {m["fn"]} @1m')
                ax.set_xlim(-51.2,51.2);ax.set_ylim(-25.6,25.6);ax.set_aspect('equal');ax.set_xlabel('Ego X (m)');ax.set_ylabel('Ego Y (m)');ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=8)
            fig.suptitle(f'{key} / validation / confidence >= 0.5 / centers bold, boundaries thin; GT endpoints circled');fig.tight_layout();fig.savefig(folder/filename,dpi=130);plt.close(fig)
        projections=[];pictures=[]
        for cam,(image,P) in enumerate(zip(rgb,mats)):
            cv2.imwrite(str(folder/f'cam{cam+1:02}.jpg'),image[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
            layers={'gt':project_segments(gt,P),'camera':project_segments(selected['camera'],P),'fusion':project_segments(selected['fusion_normal'],P)}
            a=np.c_[xyz,np.ones(len(xyz))]@P.T;good=a[:,2]>.1;uv=a[good,:2]/a[good,2:3];uv=uv[(uv[:,0]>=0)&(uv[:,0]<384)&(uv[:,1]>=0)&(uv[:,1]<384)]
            projections.append({'camera':cam+1,'image':str((folder/f'cam{cam+1:02}.jpg').relative_to(STORE)),'layers':layers,'lidar_cells_uv':uv.round(2).tolist()})
            view=image[:,:,::-1].copy()
            for name,bgr in [('gt',(90,134,37)),('camera',(200,121,40)),('fusion',(36,130,213))]:
                for part,x1,y1,x2,y2 in layers[name]:cv2.line(view,(x1,y1),(x2,y2),bgr,2 if part==0 else 1,cv2.LINE_AA)
            cv2.rectangle(view,(0,0),(145,27),(25,35,45),-1);cv2.putText(view,f'{key} / CAM{cam+1:02}',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);pictures.append(view)
        montage=np.vstack([np.hstack(pictures[:3]),np.hstack(pictures[3:])]);cv2.imwrite(str(folder/'cameras.jpg'),montage,[cv2.IMWRITE_JPEG_QUALITY,94])
        info.update({'camera_views':projections,'default_camera':max(projections,key=lambda p:len(p['layers']['gt'])+len(p['lidar_cells_uv']))['camera'],'images':{name:str((folder/name).relative_to(STORE)) for name in ['comparison.png','ablation.png','cameras.jpg']}})
        write_json(folder/'case.json',info);state('render',index+1,len(rows),frame=key)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for c in CONDITIONS:
        curve=analysis['curves'][c];x=[r['confidence'] for r in curve]
        axes[0].plot(x,[r['geometry']['1.0m']['f1'] for r in curve],marker='.',color=COLORS[c],label=c)
        axes[1].plot([r['geometry']['1.0m']['recall'] for r in curve],[r['geometry']['1.0m']['precision'] for r in curve],marker='.',color=COLORS[c],label=c)
    axes[0].set(xlabel='Confidence threshold',ylabel='Geometry F1 @1m',title='Validation threshold sweep')
    axes[1].set(xlabel='Recall @1m',ylabel='Precision @1m',title='Validation precision / recall')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'thresholds.png',dpi=150);plt.close(fig)
    labels=CONDITIONS[1:];normal=[next(r for r in analysis['curves'][c] if r['confidence']==.5)['geometry']['1.0m'] for c in labels]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(labels,[r['f1'] for r in normal],color=[COLORS[c] for c in labels]);axes[0].set(title='Same fusion checkpoint, validation F1 @1m',ylabel='F1 (confidence >= 0.5)')
    delta=[analysis['sensitivity'][c]['mean_query_center_delta_m'] for c in labels[1:]]
    axes[1].bar(['Zero vs normal','Shuffled vs normal'],delta,color=[COLORS[c] for c in labels[1:]]);axes[1].set(title='Input sensitivity (same query)',ylabel='Mean centerline XYZ change (m)')
    for ax in axes:ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'lidar-ablation.png',dpi=150);plt.close(fig)

def main():
    import fcntl
    ensure_storage();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from .dataset import Dataset
    dataset=Dataset();ids=[i for i,r in enumerate(dataset.frames) if r['split']=='val']
    assert len(ids)==47 and all(dataset.frames[i]['track']=='B' for i in ids)
    rows=[dataset.frames[i] for i in ids];donors=derangement(ids)
    manifest={'protocol':'validation-diagnostics-v1','track':'B','split':'val','frames':47,'frame_keys':[r['key'] for r in rows],'seed':42,'thresholds':THRESHOLDS,'dataset_sha256':digest(STORE/'artifacts/prepared/DATASET.json'),'checkpoint_sha256':{r:digest(STORE/'runs'/r/'best.pth') for r in ['camera-24e','fusion-24e']},'donors':{dataset.frames[i]['key']:dataset.frames[j]['key'] for i,j in donors.items()},'checkpoint_epoch':6,'threshold_selection':'validation only; no test inputs read','matching':'Same ordered 10-point mean centerline XYZ Hungarian assignment as original evaluation','limitations':['Input ablation of a trained fusion checkpoint is not a retraining comparison.','Shuffled donor frames come only from validation, with no self donor.','Projected LiDAR cells use grid center and minimum Z; they are not original LAS points.']}
    existing=read_json(OUT/'manifest.json')
    if existing and existing!=manifest:raise RuntimeError('Diagnostic cache provenance changed; use a new output version')
    write_json(OUT/'manifest.json',manifest);state('prepare',0,47)
    peak=infer(dataset,ids,donors);state('aggregate')
    analysis=summarize(rows,donors,dataset);render(rows,analysis)
    # Small index; large camera projection arrays are loaded per case by the UI.
    brief=[{k:v for k,v in case.items() if k not in ['camera_views']} for case in analysis['cases']]
    result={**manifest,**analysis,'cases':brief,'completed_at':now(),'gpu_peak_observed_mib':peak,'status':'completed','artifacts':{'thresholds':str((OUT/'thresholds.png').relative_to(STORE)),'lidar_ablation':str((OUT/'lidar-ablation.png').relative_to(STORE))}}
    assert manifest['checkpoint_sha256']=={r:digest(STORE/'runs'/r/'best.pth') for r in manifest['checkpoint_sha256']}
    write_json(OUT/'summary.json',result)
    with (OUT/'thresholds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['condition','confidence','tolerance','tp','fp','fn','predicted','gt','precision','recall','f1']);writer.writeheader()
        for c,curve in analysis['curves'].items():
            for r in curve:
                for tolerance,m in r['geometry'].items():writer.writerow(dict(condition=c,confidence=r['confidence'],tolerance=tolerance,**m))
    text='# Track B validation diagnostics\n\nValidation47 only; same epoch6 best checkpoints; no retraining and no test inputs.\n\n'
    text+='## LiDAR input ablation (confidence0.5, tolerance1m)\n\n| Condition | TP | FP | FN | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in CONDITIONS:
        m=next(r for r in analysis['curves'][c] if r['confidence']==.5)['geometry']['1.0m']
        text+=f"| {c} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |\n"
    text+='\n## Validation-selected thresholds (not held-out test performance)\n\n'
    for c in CONDITIONS:
        b=analysis['best_validation_thresholds'][c];m=b['geometry']['1.0m']
        text+=f"- {c}: confidence={b['confidence']:.2f}, F1={m['f1']:.6f}, precision={m['precision']:.6f}, recall={m['recall']:.6f}\n"
    text+='\n## Direction-free proximity (segmentation diagnostic, not instance F1)\n\n```json\n'+json.dumps(analysis['geometry_proximity'],indent=2)+'\n```\n\n'
    text+='\n## Same-query sensitivity\n\n```json\n'+json.dumps(analysis['sensitivity'],indent=2)+'\n```\n\n'
    text+='Input ablation diagnoses dependency of the already-trained fusion model; it does not establish the benefit of retraining with LiDAR. Grid-cell projections are approximate, not survey-grade calibration. Threshold matching reproduces the existing evaluator; no official OpenLane-V2 score is claimed. See manifest.json for exact validation keys, donor mapping and checkpoint hashes.\n'
    (OUT/'DIAGNOSTICS_REPORT.md').write_text(text)
    write_json(STATE,{'status':'completed','phase':'done','updated_at':now(),'completed':47,'total':47,'pid':os.getpid(),'output':str(OUT.relative_to(STORE))})
    print('DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BlockingIOError:
        print('A diagnostics worker already holds the lock; leaving it untouched.',flush=True)
    except Exception as e:
        write_json(STATE,{'status':'failed','updated_at':now(),'error':str(e),'traceback':traceback.format_exc(),'pid':os.getpid()});raise
