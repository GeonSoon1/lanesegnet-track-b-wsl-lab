import collections,json,logging,hashlib,copy
import numpy as np,torch,cv2
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from pyproj import Transformer
from .common import STORE,write_json,read_json,status,now,STORAGE
from .dataset import Dataset
from .model import LabModel,seed_all

def evaluate_run(name,dataset):
    run=STORE/'runs'/name;summary=read_json(run/'summary.json');settings=summary['settings']
    dataset.resolution=settings['resolution'];seed_all();logging.disable(logging.INFO)
    try:model=LabModel(settings['mode'],load_initial=False).cuda()
    finally:logging.disable(logging.NOTSET)
    checkpoint_path=run/'best.pth'
    digest=hashlib.sha256()
    with checkpoint_path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
    provenance={'checkpoint':str(checkpoint_path),'checkpoint_sha256':digest.hexdigest(),'checkpoint_epoch':ck['epoch'],'dataset_sha256':settings['dataset_sha256'],'settings':settings}
    del ck
    results={};world=[];features=[];fusion_sensitivity=None;transform=Transformer.from_crs(32652,4326,always_xy=True)
    for split in ['val','test']:
        counts={t:{'tp':0,'predicted':0,'gt':0} for t in [.5,1.]};errors=[];type_ok=type_n=0;top_tp=top_fp=top_fn=0;ids=[i for i,r in enumerate(dataset.frames) if r['split']==split]
        for idx in ids:
            row=dataset.frames[idx];img,metas,target,lidar=dataset.get(idx)
            with autocast():pred,rel=model.predict(img,metas,lidar if model.use_lidar else None)
            p=pred[0];keep=np.flatnonzero(p[1]>=.5);lanes=p[0][keep].reshape(-1,3,10,3);gt=target['lanes'][0].cpu().numpy().reshape(-1,3,10,3)
            if model.use_lidar and split=='val' and fusion_sensitivity is None:
                with autocast():
                    repeated,_=model.predict(img.clone(),copy.deepcopy(metas),lidar)
                    zeroed,_=model.predict(img.clone(),copy.deepcopy(metas),torch.zeros_like(lidar))
                baseline=float(np.max(np.abs(p[0]-repeated[0][0])))
                delta=float(np.max(np.abs(p[0]-zeroed[0][0])))
                fusion_sensitivity={'frame':row['key'],'split':'val','observed_lidar_cells':int((lidar[:,4]>0).sum()),'projection_weight_l2':float(model.fusion.projection.weight.detach().norm()),'repeat_max_xyz_delta_m':baseline,'zero_lidar_max_xyz_delta_m':delta,'zero_lidar_max_score_delta':float(np.max(np.abs(p[1]-zeroed[0][1]))),'effect_exceeds_repeat_noise':delta>max(1e-5,10*baseline),'scope':'One validation input ablation, no retraining or model/threshold selection. Not evidence of accuracy improvement.'}
                del repeated,zeroed
            if len(lanes) and len(gt):
                distance=np.linalg.norm(lanes[:,None,0,:,:]-gt[None,:,0,:,:],axis=-1).mean(-1);pi,gi=linear_sum_assignment(distance)
            else:distance=np.empty((len(lanes),len(gt)));pi=np.array([],int);gi=np.array([],int)
            for threshold,c in counts.items():c['tp']+=int(sum(distance[a,b]<=threshold for a,b in zip(pi,gi)));c['predicted']+=len(lanes);c['gt']+=len(gt)
            valid=[(a,b) for a,b in zip(pi,gi) if distance[a,b]<=1.]
            for a,b in valid:
                errors.append(float(distance[a,b]))
                for resultindex,key in [(4,'left_types'),(6,'right_types')]:
                    truth=int(target[key][0][b]);
                    if truth>=0:type_n+=1;type_ok+=int(int(p[resultindex][keep[a]])==truth)
            if valid:
                pp=[keep[a] for a,b in valid];gg=[b for a,b in valid];prediction=rel[0].cpu().numpy()[np.ix_(pp,pp)]>=.5;truth=target['adjacency'][0].cpu().numpy()[np.ix_(gg,gg)]>0
                mask=~np.eye(len(pp),dtype=bool);top_tp+=int((prediction&truth&mask).sum());top_fp+=int((prediction&~truth&mask).sum());top_fn+=int((~prediction&truth&mask).sum())
            R=np.array(row['rotation']);origin=np.array(row['origin']);wl=lanes@R.T+origin
            world.append({'key':row['key'],'split':split,'timestamp':row['timestamp'],'scores':p[1][keep].tolist(),'lanes_center_left_right_xyz':wl.tolist(),'crs':'EPSG:32652','left_types':p[4][keep].tolist(),'right_types':p[6][keep].tolist()})
            for lane,score in zip(wl,p[1][keep]):
                lon,lat=transform.transform(lane[0,:,0],lane[0,:,1]);features.append({'type':'Feature','geometry':{'type':'LineString','coordinates':list(map(list,zip(lon,lat)))},'properties':{'frame':row['key'],'score':float(score),'model':name,'split':split}})
        measures={}
        for threshold,c in counts.items():
            precision=c['tp']/max(1,c['predicted']);recall=c['tp']/max(1,c['gt']);measures[str(threshold)+'m']=dict(c,precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall))
        results[split]={'frames':len(ids),'geometry':measures,'mean_matched_center_xyz_error_m':float(np.mean(errors)) if errors else None,'type_accuracy_on_matched_known':type_ok/type_n if type_n else None,'type_observations':type_n,'topology_on_matched_pairs':{'tp':top_tp,'fp':top_fp,'fn':top_fn}}
    write_json(run/'evaluation.json',{'results':results,'provenance':provenance,'fusion_validation_sensitivity':fusion_sensitivity,'confidence_threshold':.5,'distance':'Ordered mean centerline XYZ distance,10 samples','official_openlanev2_score':False,'topology_scope':'Only pairs of lanes matched within1m; not official TOP metric'})
    write_json(run/'world-predictions.json',{'crs':'EPSG:32652','frames':world,'provenance':provenance})
    write_json(run/'predictions-wgs84.geojson',{'type':'FeatureCollection','features':features})
    del model;torch.cuda.empty_cache();return results

def main():
    status('evaluation',message='Evaluating both best checkpoints on identical held-out spatial regions')
    dataset=Dataset();results={}
    for name in [STORAGE.get('run_prefix','')+m+'-24e' for m in ['camera','fusion']]:results[name]=evaluate_run(name,dataset)
    write_json(STORE/'reports/experiment-comparison.json',results)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(results);fig,ax=plt.subplots(figsize=(7,4));x=np.arange(2)
    for i,name in enumerate(names):ax.bar(x+(i-.5)*.32,[results[name][s]['geometry']['1.0m']['f1'] for s in ['val','test']],.32,label=name)
    ax.set_xticks(x,['Validation','Held-out test']);ax.set_ylabel('Geometry F1 @1m, confidence>=0.5');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(STORE/'reports/comparison.png',dpi=150);plt.close(fig)
    text='# GG49 training results\n\nBoth models used the same official initial weights, fixed frames and spatial split. All results are measured on this local GG49 split, not official OpenLane-V2 benchmark scores.\n\n'
    for name in names:
        summary=read_json(STORE/'runs'/name/'summary.json');text+=f"## {name}\n\nEpochs: {summary['epoch']}; best validation loss: {summary['best_val']:.6f}.\n\n```json\n{json.dumps(results[name],indent=2)}\n```\n\n"
    text+='LiDAR benefit must be judged from the comparison; completing training alone does not prove improvement. Map stitching remains a subsequent stage.\n'
    (STORE/'reports/TRAINING_RESULTS.md').write_text(text)
    # Complete only after both real runs and their durable artifacts exist.
    for name in names:
        summary=read_json(STORE/'runs'/name/'summary.json')
        assert summary.get('status')=='completed' and summary.get('epoch')==24
        for artifact in ['best.pth','last.pth','loss.png','history.csv','evaluation.json']:
            assert (STORE/'runs'/name/artifact).stat().st_size>0
    implementation=read_json(STORE/'reports/IMPLEMENTATION_STATUS.json')
    implementation.update(implementation='completed',training_complete=True,completed_at=now(),comparison='reports/experiment-comparison.json',results=results,current={'phase':'done','status':'completed','updated_at':now()})
    write_json(STORE/'reports/IMPLEMENTATION_STATUS.json',implementation)
    # Measured outputs stay in reports; design PRD and retrospective are maintained separately.
    status('evaluation','completed','Both24-epoch experiments, checkpoints, graphs and held-out evaluation completed',comparison='reports/experiment-comparison.json')
if __name__=='__main__':main()
