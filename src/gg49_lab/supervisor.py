"""Persistent stage supervisor. Original data is read-only; products stay on D:."""
import json,os,subprocess,sys,time,traceback
from .common import ROOT,STORE,PYTHON,status,read_json,write_json,now,STORAGE
PREFIX=STORAGE.get("run_prefix", "")

def worker(module,args=(),label='worker',attempts=1):
    if (STORE/'runs/STOP').exists():raise RuntimeError('Checkpointed stop requested; no next stage launched')
    logs=STORE/'logs';logs.mkdir(exist_ok=True)
    for attempt in range(attempts):
        with (logs/(label+'.log')).open('ab') as f:
            p=subprocess.Popen([PYTHON,'-u','-m',module,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            write_json(STORE/'runs/active-worker.json',{'pid':p.pid,'module':module,'args':list(map(str,args)),'started_at':now(),'log':str(logs/(label+'.log'))})
            while p.poll() is None:
                time.sleep(10)
                state=read_json(STORE/'runs/pipeline-status.json');state['heartbeat']=now();write_json(STORE/'runs/pipeline-status.json',state)
            if p.returncode==0:return
        print('Worker failed',label,'attempt',attempt+1,flush=True)
        if (STORE/'runs/STOP').exists():raise RuntimeError('Checkpointed stop requested; no retry launched')
        if attempt+1<attempts:time.sleep(5)
    raise RuntimeError(f'{label} failed; inspect {logs/(label+".log")}')

def main():
    status('environment',message='Local isolated Python environment; original environments preserved')
    # A completed preparation cache is reusable across process restarts.
    if not (STORE/'artifacts/prepared/DATASET.json').exists():worker('gg49_lab.prepare',label='prepare')
    status('model',message='Real LaneSegNet camera model, masked type loss, directed topology and LiDAR fusion adapter ready')
    settings=read_json(STORE/'reports/memory-selected.json')
    if not settings.get('passed'):
        status('tune',message='Testing real forward/backward and optimizer steps against the12GB GPU budget')
        for resolution in [384,320,256]:
            for batch in [2,1]:
                ok=True
                for mode in ['fusion','camera']:
                    output=STORE/'reports'/f'probe-{mode}-{resolution}-b{batch}.json'
                    try:worker('gg49_lab.train',['probe','--mode',mode,'--resolution',resolution,'--batch',batch,'--output',output],label=f'probe-{mode}-{resolution}-b{batch}')
                    except RuntimeError:
                        report=read_json(output)
                        error=report.get('error','').lower()
                        if 'out of memory' not in error and 'memory headroom' not in error:raise
                        ok=False;break
                    if read_json(output).get('status')!='passed':ok=False;break
                if ok:settings={'passed':True,'resolution':resolution,'micro_batch':batch,'effective_batch':4,'global_gpu_budget_mib':10752};break
            if settings.get('passed'):break
        if not settings.get('passed'):raise RuntimeError('No tested configuration passed; no full training launched')
        write_json(STORE/'reports/memory-selected.json',settings)
    demo=PREFIX+'demo-camera-1e'
    if read_json(STORE/'runs'/demo/'summary.json').get('status')!='completed':worker('gg49_lab.train',['train','--mode','camera','--resolution',settings['resolution'],'--batch',settings['micro_batch'],'--run',demo,'--epochs',1],label=demo)
    for mode in ['camera','fusion']:
        run=PREFIX+'pilot-'+mode
        if read_json(STORE/'runs'/run/'summary.json').get('status')!='completed':worker('gg49_lab.train',['train','--mode',mode,'--resolution',settings['resolution'],'--batch',settings['micro_batch'],'--run',run,'--pilot-steps',100],label=run)
        pilot=read_json(STORE/'runs'/run/'summary.json')
        if not pilot.get('learning_response'):raise RuntimeError(mode+' pilot loss did not decline; inspect before full training')
    for mode in ['camera','fusion']:
        run=PREFIX+mode+'-24e'
        if read_json(STORE/'runs'/run/'summary.json').get('status')!='completed':worker('gg49_lab.train',['train','--mode',mode,'--resolution',settings['resolution'],'--batch',settings['micro_batch'],'--run',run,'--epochs',24],label=run,attempts=2)
    worker('gg49_lab.evaluate',label='evaluation')
    status('export',message='Exporting new held-out predictions to verified XYZ SHP')
    for mode in ['camera','fusion']:
        worker('gg49_lab.export_local',['--run',PREFIX+mode+'-24e'],label='export-'+mode)
    status('done','completed','New 24-epoch comparison, held-out evaluation and verified SHP ZIP exports completed')
if __name__=='__main__':
    try:main()
    except Exception as e:
        traceback.print_exc();status(read_json(STORE/'runs/pipeline-status.json').get('phase','pipeline'),'interrupted' if (STORE/'runs/STOP').exists() else 'failed',str(e));sys.exit(1)
