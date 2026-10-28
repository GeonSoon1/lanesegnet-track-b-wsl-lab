#!/usr/bin/env python3
from pathlib import Path
import sys,json,subprocess,urllib.parse,mimetypes,os
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gg49_lab.common import STORE,read_json
def current_state(data,run=None):
    data=dict(data)
    pid=data.get('pid')
    if data.get('status') in ['running','validating']:
        try:
            cmd=Path(f'/proc/{int(pid)}/cmdline').read_bytes()
            active=b'gg49_lab' in cmd or b'local_pipeline' in cmd
            if run is not None:active=active and (b'--run\x00'+run.encode()+b'\x00') in cmd
        except (OSError,TypeError,ValueError):active=False
        data['process_alive']=active
        if not active:data['status']='interrupted'
    return data
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,body,ctype='application/json',code=200):
        data=body if isinstance(body,bytes) else body.encode('utf-8')
        self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def json(self,obj,code=200):self.send(json.dumps(obj,ensure_ascii=False),code=code)
    def do_GET(self):
        path=urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        if path=='/':return self.send((ROOT/'dashboard/index.html').read_bytes(),'text/html; charset=utf-8')
        if path in ['/experiments','/experiments/']:
            return self.send((ROOT/'dashboard/experiments.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/assets/experiments.js':
            return self.send((ROOT/'dashboard/experiments.js').read_bytes(),'application/javascript; charset=utf-8')
        if path=='/api/followup':
            return self.json({'status':read_json(STORE/'runs/followup-status.json'),'summary':read_json(STORE/'reports/followup-45-v1/summary.json')})
        if path in ['/diagnostics','/diagnostics/']:
            return self.send((ROOT/'dashboard/diagnostics.html').read_bytes(),'text/html; charset=utf-8')
        if path in ['/assets/diagnostics.css','/assets/diagnostics.js']:
            asset=ROOT/'dashboard'/path.split('/')[-1]
            return self.send(asset.read_bytes(),'text/css; charset=utf-8' if asset.suffix=='.css' else 'application/javascript; charset=utf-8')
        if path=='/api/diagnostics':
            log=STORE/'logs/diagnostics.log';tail=[]
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-6000));tail=f.read().decode('utf-8','replace').splitlines()[-20:]
            return self.json({'status':read_json(STORE/'runs/diagnostics-status.json'),'summary':read_json(STORE/'reports/diagnostics-track-b-v1/summary.json'),'log_tail':tail})
        if path=='/api/status':
            d=current_state(read_json(STORE/'runs/pipeline-status.json'));d['storage_windows']='D:/mms_experiments/gg49-lanesegnet-lab_gs'
            if d.get('phase') in ['camera','fusion','pilot'] and d.get('current_run'):
                r=current_state(read_json(STORE/'runs'/d['current_run']/'summary.json'),d['current_run'])
                if r.get('pid')==d.get('pid') and r.get('status'):d['status']=r['status']
            try:
                out=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
                d['gpu']={'name':out[0].strip(),'used_mib':int(out[1]),'total_mib':int(out[2]),'utilization':int(out[3])}
            except Exception:d['gpu']=None
            active=read_json(STORE/'runs/active-worker.json');log=Path(active['log']) if active.get('log') else STORE/'logs/pipeline.log'
            if log.exists():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-20000));d['log_tail']=f.read().decode('utf-8','replace').splitlines()[-35:]
            return self.json(d)
        if path=='/api/runs':
            return self.json([dict(id=p.parent.name,**current_state(read_json(p),p.parent.name)) for p in sorted((STORE/'runs').glob('*/summary.json'))])
        if path.startswith('/api/runs/') and path.endswith('/metrics'):
            run=path.split('/')[3]
            if not run.replace('-','').replace('_','').isalnum():return self.json({'error':'invalid run'},400)
            p=STORE/'runs'/run/'metrics.jsonl';rows=[]
            if p.exists():
                for line in p.read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
            latest={}
            for row in rows:latest[(row.get('kind'),row.get('step') if row.get('kind')=='step' else row.get('epoch'))]=row
            return self.json(list(latest.values()))
        if path.startswith('/files/'):
            p=(STORE/path[len('/files/'):]).resolve()
            if not p.is_relative_to(STORE.resolve()) or p.suffix.lower() not in ['.png','.jpg','.svg','.json','.csv','.md','.txt','.geojson','.gpkg','.zip','.shp','.shx','.dbf','.prj','.cpg']:return self.json({'error':'not allowed'},403)
            if p.is_file():return self.send(p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
        return self.json({'error':'not found'},404)
if __name__=='__main__':
    print('DASHBOARD http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('0.0.0.0' if '--wsl-host' in sys.argv else '127.0.0.1',8765),Handler).serve_forever()
