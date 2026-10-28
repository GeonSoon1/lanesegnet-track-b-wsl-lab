"""Sequential, locked replay of experiment4/5. Never starts training."""
import fcntl,json,os,subprocess,sys
from .common import STORE,ROOT,read_json,write_json,now
from .reference_experiment import OUT,sha

def main():
 lock=(OUT/'pipeline.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Existing followup pipeline is running; not duplicated.');return
 result=read_json(OUT/'summary.json');old=read_json(STORE/'runs/followup-status.json')
 if old.get('status')=='running' and old.get('pid')!=os.getpid():
  p=__import__('pathlib').Path('/proc')/str(old.get('pid'))/'cmdline'
  if p.exists() and b'gg49_lab.' in p.read_bytes():raise RuntimeError('An existing experiment worker is alive; not duplicated')
 if result:
  provenance=result['reference']['provenance']
  assert sha(STORE/'artifacts/prepared/DATASET.json')==provenance['dataset_sha256']
  assert all(sha(STORE/'runs'/f'{mode}-24e/best.pth')==value for mode,value in provenance['checkpoint_sha256'].items())
  assert all(sha(ROOT/p)==value for p,value in provenance['inference_source_sha256'].items())
  if '--rerun' not in sys.argv:
   print('Completed experiment artifacts preserved. Open http://127.0.0.1:8765/experiments . Use --rerun only for explicit replay.');return
 for module in ['reference_experiment','map_experiment','support_experiment','followup_visuals']:
  with (STORE/'logs'/f'followup-{module}.log').open('ab') as log:subprocess.run([sys.executable,'-u','-m','gg49_lab.'+module],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
