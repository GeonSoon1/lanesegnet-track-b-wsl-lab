"""Actual GPU training, VRAM probing, checkpoints, validation and live metrics."""
import argparse,collections,gc,json,logging,math,os,random,subprocess,time,traceback,uuid
from pathlib import Path
import numpy as np,torch
from torch.cuda.amp import autocast,GradScaler
from torch.utils.tensorboard import SummaryWriter
from .common import ROOT,STORE,now,read_json,write_json,status,append_jsonl,ensure_storage
from .dataset import Dataset
from .model import LabModel,seed_all

def build(mode):
    seed_all();logging.disable(logging.INFO)
    try:model=LabModel(mode).cuda()
    finally:logging.disable(logging.NOTSET)
    seed_all();return model

def losses_total(losses):
    terms={k:(sum(v) if isinstance(v,(list,tuple)) else v).mean() for k,v in losses.items() if 'loss' in k}
    total=sum(terms.values())
    if not torch.isfinite(total):raise FloatingPointError('Non-finite loss: '+str({k:float(v.detach()) for k,v in terms.items()}))
    return total,terms

def optimizer_for(model):
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('net.img_backbone')],'lr':2e-6},{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('net.img_backbone')],'lr':2e-5}]
    return torch.optim.AdamW(groups,lr=2e-5,weight_decay=.01)

def update(model,dataset,micro_batches,optimizer,scaler):
    size=sum(len(x) for x in micro_batches)
    for retry in range(5):
        optimizer.zero_grad(set_to_none=True);tot=0.;components=collections.defaultdict(float)
        for ids in micro_batches:
            img,meta,target,lidar=dataset.batch(ids)
            with autocast():
                total,parts=losses_total(model(img,meta,target,lidar if model.use_lidar else None));weighted=total*len(ids)/size
            scaler.scale(weighted).backward()
            tot+=float(total.detach())*len(ids)/size
            for k,v in parts.items():components[k]+=float(v.detach())*len(ids)/size
            del img,meta,target,lidar,total,parts,weighted
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if torch.isfinite(norm):
            scaler.step(optimizer);scaler.update();return tot,dict(components),float(norm),retry
        scaler.update(new_scale=max(scaler.get_scale()/2,1.))
        print('AMP gradient overflow: retry scale',scaler.get_scale(),flush=True)
    raise FloatingPointError('Gradient remained non-finite after AMP rescaling')

@torch.no_grad()
def validate(model,dataset,ids):
    model.eval();parts=collections.defaultdict(float);total=0.
    for idx in ids:
        img,meta,target,lidar=dataset.get(idx)
        with autocast():value,loss=losses_total(model(img,meta,target,lidar if model.use_lidar else None))
        total+=float(value)
        for k,v in loss.items():parts[k]+=float(v)
    return total/len(ids),{k:v/len(ids) for k,v in parts.items()}

def gpu_used():
    try:return int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0])
    except Exception:return 0

def probe(args):
    dataset=Dataset(args.resolution);ids=sorted([i for i,r in enumerate(dataset.frames) if r['split']=='train'],key=lambda i:dataset.frames[i]['lanes'],reverse=True)[:max(4,args.batch*2)]
    model=build(args.mode);model.train();optimizer=optimizer_for(model);scaler=GradScaler(init_scale=128)
    torch.cuda.reset_peak_memory_stats();start=time.time();records=[]
    before=model.net.pts_bbox_head.reg_branches[0][0].weight.detach().clone()
    for turn in range(2):
        chosen=ids[turn*args.batch:(turn+1)*args.batch]
        value,parts,norm,retries=update(model,dataset,[chosen],optimizer,scaler)
        torch.cuda.synchronize();records.append({'loss':value,'gradient_norm':norm,'global_gpu_used_mib':gpu_used()})
    delta=float((model.net.pts_bbox_head.reg_branches[0][0].weight.detach()-before).abs().max());assert delta>0
    if model.use_lidar:
        assert model.fusion.projection.weight.grad is not None and model.fusion.projection.weight.grad.abs().sum()>0
        assert model.fusion.encoder[0].weight.grad is not None and model.fusion.encoder[0].weight.grad.abs().sum()>0
    peak=max(r['global_gpu_used_mib'] for r in records)
    result={'status':'passed' if peak<=10752 else 'over_budget','mode':args.mode,'resolution':args.resolution,'batch':args.batch,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'global_gpu_used_mib':peak,'seconds':time.time()-start,'weight_max_delta':delta,'records':records,'lidar_encoder_gradient_verified':model.use_lidar}
    write_json(args.output,result);print('PROBE',json.dumps(result),flush=True)
    if result['status']!='passed':raise RuntimeError('GPU memory headroom below1.5GiB')

def checkpoint(path,model,optimizer,scaler,epoch,step,order,position,settings,best):
    obj={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'step':step,'order':order,'position':position,'settings':settings,'best_val':best,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'scheduler':{'kind':'step-derived-warmup-cosine','optimizer_step':step}}
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(obj,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)
    check=torch.load(path,map_location='cpu');assert check['step']==step and 'optimizer' in check;del check

def plot_history(run):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for line in (run/'metrics.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    latest={}
    for row in rows:latest[(row.get('kind'),row.get('step') if row.get('kind')=='step' else row.get('epoch'))]=row
    rows=list(latest.values())
    steps=[x for x in rows if x['kind']=='step'];epochs=[x for x in rows if x['kind']=='epoch']
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].plot([x['step'] for x in steps],[x['total'] for x in steps],lw=.7,color='#377ec0');axs[0].set(xlabel='Optimizer step',ylabel='Weighted total loss',title='Training loss')
    if epochs:
        axs[1].plot([x['epoch'] for x in epochs],[x['train_total'] for x in epochs],label='train');axs[1].plot([x['epoch'] for x in epochs],[x['val_total'] for x in epochs],label='validation');axs[1].legend()
    axs[1].set(xlabel='Epoch',ylabel='Weighted total loss',title='Same fixed spatial validation split')
    for ax in axs:ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(run/'loss.png',dpi=150);plt.close(fig)
    import csv
    keys=['step','epoch','kind','total','train_total','val_total','seconds']
    with (run/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def preview(model,dataset,idx,run,epoch):
    import cv2
    model.eval();img,meta,target,lidar=dataset.get(idx)
    with autocast():pred,_=model.predict(img,meta,lidar if model.use_lidar else None)
    lanes,scores,labels=pred[0][:3];selected=lanes[scores>=.5].reshape(-1,3,10,3)
    gt=target['lanes'][0].cpu().numpy().reshape(-1,3,10,3)
    canvas=np.full((540,1060,3),250,np.uint8)
    for col,(items,title,color) in enumerate([(gt,'SHP ground truth',(150,70,150)),(selected,'Predicted lanes: score >= 0.5',(60,140,220))]):
        x0=col*530+20;y0=70;w=490;h=400
        cv2.putText(canvas,title,(x0,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,45,60),1)
        for lane in items:
            for k in range(3):
                uv=np.column_stack(((lane[k,:,0]+51.2)/102.4*w+x0,(25.6-lane[k,:,1])/51.2*h+y0)).round().clip(-10000,10000).astype(np.int32)
                cv2.polylines(canvas,[uv],False,color,2 if k==0 else 1,cv2.LINE_AA)
        cv2.rectangle(canvas,(x0,y0),(x0+w,y0+h),(190,200,210),1)
    cv2.putText(canvas,f"{dataset.frames[idx]['key']} | epoch {epoch} | {model.mode}",(20,520),cv2.FONT_HERSHEY_SIMPLEX,.6,(40,55,70),1)
    p=run/'prediction.png';cv2.imwrite(str(p),canvas);return str(p.relative_to(STORE))

def train(args):
    if not args.run.replace('-','').replace('_','').isalnum():raise ValueError('Run must be a single experiment name without paths')
    torch.set_num_threads(4);dataset=Dataset(args.resolution);trainids=[i for i,r in enumerate(dataset.frames) if r['split']=='train'];valids=[i for i,r in enumerate(dataset.frames) if r['split']=='val']
    pilot=args.pilot_steps>0
    if pilot:trainids=trainids[:16]
    run=STORE/'runs'/args.run;run.mkdir(parents=True,exist_ok=True)
    settings={'mode':args.mode,'use_camera':True,'use_lidar':args.mode=='fusion','epochs':0 if pilot else args.epochs,'pilot_steps':args.pilot_steps,'resolution':args.resolution,'micro_batch':args.batch,'effective_batch':4,'lr':2e-5,'backbone_lr':2e-6,'seed':42,'environment':str(__import__('sys').executable),'reference_policy':'aligned-v1','initial_checkpoint_sha256':__import__('hashlib').sha256((STORE/'checkpoints/lanesegnet_r50_8x1_24e_olv2_subset_A.pth').read_bytes()).hexdigest(),'train_frames':len(trainids),'val_frames':len(valids),'dataset_sha256':__import__('hashlib').sha256((STORE/'artifacts/prepared/DATASET.json').read_bytes()).hexdigest()}
    summary=read_json(run/'summary.json')
    for key in ['error','traceback']:summary.pop(key,None)
    summary.update(pid=os.getpid(),status='running',modality=args.mode,kind='pilot' if pilot else 'training',settings=settings,started_at=summary.get('started_at',now()),epoch=summary.get('epoch',0))
    write_json(run/'summary.json',summary);status('pilot' if pilot else args.mode,message='Actual GPU training and validation',current_run=args.run)
    model=build(args.mode);optimizer=optimizer_for(model);scaler=GradScaler(init_scale=128);start_epoch=0;step=0;best=float('inf');position=0;resume_order=None
    if (run/'last.pth').exists():
        ck=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        if ck['settings']!=settings:raise RuntimeError('Resume settings differ; choose a fresh run')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start_epoch=ck['epoch'];step=ck['step'];best=ck['best_val'];position=ck['position'];resume_order=ck['order']
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state_all(ck['rng_cuda']);del ck
        append_jsonl(run/'resume-events.jsonl',{'loaded_step':step,'epoch_zero_based':start_epoch,'data_position':position,'pid':os.getpid(),'note':'Updates after this checkpoint were not restored; any later steps are recomputed.'})
    writer=SummaryWriter(str(run/'tensorboard'),purge_step=step+1 if step else None);begin=time.time();history=[]
    epochs=10000 if pilot else args.epochs
    for epoch in range(start_epoch,epochs):
        model.train();order=resume_order if resume_order is not None else trainids.copy()
        if resume_order is None:random.Random(42+epoch).shuffle(order)
        resume_order=None;epoch_values=[];epoch_parts=collections.defaultdict(list)
        if position>0 and (run/'metrics.jsonl').exists():
            prior={}
            for line in (run/'metrics.jsonl').read_text().splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if record.get('kind')=='step' and record.get('epoch')==epoch+1 and record.get('step',0)<=step:prior[record['step']]=record
            for record in prior.values():
                epoch_values.append(record['total'])
                for k,v in record['losses'].items():epoch_parts[k].append(v)
        for pos in range(position,len(order),4):
            ids=order[pos:pos+4];batches=[ids[j:j+args.batch] for j in range(0,len(ids),args.batch)]
            if not pilot:
                planned=max(1,math.ceil(len(trainids)/4)*args.epochs);ratio=min(1,(step+1)/100) if step<100 else .001+.999*(1+math.cos(math.pi*min(1,(step-100)/max(1,planned-100))))/2
                optimizer.param_groups[0]['lr']=2e-6*ratio;optimizer.param_groups[1]['lr']=2e-5*ratio
            tick=time.time();value,parts,norm,retries=update(model,dataset,batches,optimizer,scaler);torch.cuda.synchronize();elapsed=time.time()-tick;step+=1;history.append(value);epoch_values.append(value)
            for k,v in parts.items():epoch_parts[k].append(v);writer.add_scalar('train/'+k,v,step)
            row={'kind':'step','step':step,'epoch':epoch+1,'total':value,'losses':parts,'seconds':elapsed,'gradient_norm':norm,'amp_scale':scaler.get_scale(),'amp_retries':retries,'actual_batch_samples':len(ids),'gradient_accumulation_micro_batches':len(batches),'global_gpu_used_mib':gpu_used(),'lr':optimizer.param_groups[1]['lr'],'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            append_jsonl(run/'metrics.jsonl',row);writer.add_scalar('train/total',value,step)
            summary.update(step=step,last_train_loss=value,last_step_seconds=elapsed,updated_at=now(),epoch=0 if pilot else epoch)
            write_json(run/'summary.json',summary)
            status('pilot' if pilot else args.mode,message=f"{args.run}: optimizer step {step}, loss {value:.4f}",current_run=args.run)
            if (STORE/'runs/STOP').exists():
                checkpoint(run/'last.pth',model,optimizer,scaler,epoch,step,order,pos+4,settings,best)
                summary.update(status='interrupted',saved_step=step);write_json(run/'summary.json',summary);writer.close();raise RuntimeError('User requested checkpointed stop')
            if step%25==0 and not pilot:checkpoint(run/'last.pth',model,optimizer,scaler,epoch,step,order,pos+4,settings,best)
            if pilot and step>=args.pilot_steps:break
        position=0
        if pilot:
            if step>=args.pilot_steps:break
            continue
        trainvalue=float(np.mean(epoch_values)) if epoch_values else summary.get('last_train_loss',0)
        checkpoint(run/'last.pth',model,optimizer,scaler,epoch,step,order,len(order),settings,best)
        summary.update(status='validating');write_json(run/'summary.json',summary)
        valvalue,valparts=validate(model,dataset,valids)
        summary.update(status='running')
        trainparts={k:float(np.mean(v)) for k,v in epoch_parts.items()}
        record={'kind':'epoch','epoch':epoch+1,'step':step,'train_total':trainvalue,'val_total':valvalue,'train_losses':trainparts,'val_losses':valparts,'seconds':time.time()-begin}
        append_jsonl(run/'metrics.jsonl',record);writer.add_scalar('epoch/train_total',trainvalue,epoch+1);writer.add_scalar('epoch/val_total',valvalue,epoch+1)
        for k,v in valparts.items():writer.add_scalar('validation/'+k,v,epoch+1)
        if valvalue<best:
            best=valvalue;tmp=run/('best.'+uuid.uuid4().hex+'.tmp');torch.save({'model':model.state_dict(),'epoch':epoch+1,'step':step,'settings':settings,'val_total':best},tmp)
            with tmp.open('rb') as f:os.fsync(f.fileno())
            os.replace(tmp,run/'best.pth');verified=torch.load(run/'best.pth',map_location='cpu');assert verified['epoch']==epoch+1;del verified
        checkpoint(run/'last.pth',model,optimizer,scaler,epoch+1,step,None,0,settings,best)
        image=preview(model,dataset,valids[len(valids)//2],run,epoch+1)
        summary.update(epoch=epoch+1,best_epoch=(epoch+1 if valvalue==best else summary.get('best_epoch')),best_val=best,last_val=valvalue,previews=[image,'runs/'+args.run+'/loss.png'],updated_at=now());write_json(run/'summary.json',summary)
        plot_history(run);writer.flush();print('EPOCH',args.run,epoch+1,trainvalue,valvalue,flush=True)
    if pilot:
        first=float(np.median(history[:10]));last=float(np.median(history[-10:]));summary.update(pilot_first_loss=first,pilot_last_loss=last,learning_response=last<first,pilot_weight_update_verified=True)
        # Preserve measured response rather than inventing improvement.
        if not last<first:print('PILOT_WARNING: median loss did not decline; inspect figures',flush=True)
    summary.update(status='completed',completed_at=now(),step=step);write_json(run/'summary.json',summary);plot_history(run);writer.close()
    print('TRAINING_COMPLETE',args.run,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);ap.add_argument('--mode',choices=['camera','fusion'],required=True);ap.add_argument('--resolution',type=int,default=384);ap.add_argument('--batch',type=int,default=1);ap.add_argument('--run',default='camera-24e');ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--pilot-steps',type=int,default=0);ap.add_argument('--output',type=Path,default=STORE/'reports/probe.json');args=ap.parse_args()
    if args.action=='train' and not args.run.replace('-','').replace('_','').isalnum():ap.error('--run must be a single experiment name without paths')
    # Preflight before the error reporter attempts to write anything on D:.
    try:ensure_storage()
    except RuntimeError as e:ap.exit(2,str(e)+'\nMount D: in WSL before starting training.\n')
    try:
        torch.set_num_threads(4)
        if args.action=='train' and (STORE/'runs/STOP').exists():raise RuntimeError('Checkpointed stop requested; no training update launched')
        if args.action=='probe':probe(args)
        else:train(args)
    except Exception as e:
        result={'status':'failed','error':str(e),'traceback':traceback.format_exc(),'time':now(),'mode':args.mode,'resolution':args.resolution,'batch':args.batch}
        if args.action=='probe':write_json(args.output,result)
        else:
            p=STORE/'runs'/args.run/'summary.json';summary=read_json(p);summary.update(result);summary['status']='interrupted' if (STORE/'runs/STOP').exists() else 'failed';write_json(p,summary)
        traceback.print_exc();raise
if __name__=='__main__':main()
