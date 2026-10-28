"""Local-only detached process control. Does not import the training environment."""
import argparse,fcntl,json,os,subprocess,time,hashlib,shutil,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=Path('/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj')
def alive(pid,key=None):
 try:
  text=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  expected={'supervisor':'gg49_lab.supervisor','dashboard':'dashboard_server.py','prepare':'gg49_lab.prepare'}.get(key)
  return (expected in text and str(ROOT) in text) if expected else (str(ROOT) in text or 'gg49_lab' in text)
 except (OSError,ValueError):return False

def active_worker_alive():
 try:
  record=json.loads((S/'runs/active-worker.json').read_text())
  args=[x.decode() for x in Path('/proc/'+str(record['pid'])+'/cmdline').read_bytes().split(b'\0') if x]
  index=args.index('-m')
  return str(ROOT) in args[0] and args[index+1:]==[record['module'],*record['args']]
 except (OSError,ValueError,KeyError):return False
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['start','resume','status','stop','dashboard','dashboard-stop','prepare']);a=ap.parse_args()
 if not os.path.ismount('/mnt/d'):raise SystemExit('SSD is not mounted; refusing fallback.')
 (S/'runs').mkdir(exist_ok=True);(S/'logs').mkdir(exist_ok=True)
 if a.action=='dashboard-stop':
  p=S/'runs/dashboard.pid';pid=int(p.read_text()) if p.exists() else 0
  if alive(pid,'dashboard'):os.kill(pid,signal.SIGTERM);print('Dashboard stopped; training is independent and continues.')
  else:print('Dashboard is not running.')
  return
 if a.action=='stop':
  (S/'runs/STOP').write_text('Stop at next safe checkpoint.\n');print('Stop requested; next optimizer-step boundary will be saved.');return
 if a.action=='status':
  for key in ['supervisor','dashboard','prepare']:
   p=S/'runs'/(key+'.pid');pid=int(p.read_text()) if p.exists() else 0;print(key,pid,'alive' if alive(pid,key) else 'not running')
  print((S/'runs/pipeline-status.json').read_text() if (S/'runs/pipeline-status.json').exists() else 'No experiment measurements yet.');return
 key='dashboard' if a.action=='dashboard' else 'prepare' if a.action=='prepare' else 'supervisor'
 with (S/'runs'/('dashboard-launch.lock' if key=='dashboard' else 'work-launch.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=S/'runs'/(key+'.pid');old=int(p.read_text()) if p.exists() else 0
  if alive(old,key):print('Already running',key,old);return
  if key!='dashboard':
   other='prepare' if key=='supervisor' else 'supervisor';other_path=S/'runs'/(other+'.pid')
   if (other_path.exists() and alive(int(other_path.read_text()),other)) or active_worker_alive():
    raise SystemExit('A real preparation/training worker is still running; refusing duplicate work. Check status/logs, then resume after it exits.')
  if key!='dashboard' and shutil.disk_usage(S).free<20*1024**3:raise SystemExit('SSD has less than20GiB free; refusing new training/preparation.')
  probe=S/'logs'/('.write-probe-'+str(os.getpid()));probe.write_text('SSD write check');probe.unlink()
  if key=='supervisor':
   audit=S/'reports/LOCAL_COMPLETION_AUDIT.json'
   if audit.exists() and json.loads(audit.read_text()).get('status')=='verified_complete':print('This experiment is verified complete; use a new experiment name for new training.');return
   if not (S/'reports/QA_APPROVED.json').exists():raise SystemExit('Input visual QA has not been approved. Run prepare and inspect images first.')
   qa=json.loads((S/'reports/QA_APPROVED.json').read_text())
   actual=hashlib.sha256((S/'artifacts/prepared/DATASET.json').read_bytes()).hexdigest()
   if qa.get('dataset_sha256')!=actual:raise SystemExit('Dataset changed since visual QA; inspect and update the explicit QA record before launch.')
   (S/'runs/STOP').unlink(missing_ok=True)
  command={'dashboard':f'exec python3 "{ROOT}/scripts/dashboard_server.py"','prepare':'exec "$PY" -u -m gg49_lab.prepare','supervisor':f'exec bash "{ROOT}/scripts/local_pipeline.sh"'}[key]
  with (S/'logs'/(key+'.log')).open('ab') as log:
   worker=subprocess.Popen(['bash','-c',f'source "{ROOT}/scripts/local_env.sh"; {command}'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  p.write_text(str(worker.pid));time.sleep(.5)
  if not alive(worker.pid,key):raise SystemExit('Process did not start; inspect '+str(S/'logs'/(key+'.log')))
  print(key,'PID',worker.pid,'alive',True,'log',S/'logs'/(key+'.log'))
if __name__=='__main__':main()
