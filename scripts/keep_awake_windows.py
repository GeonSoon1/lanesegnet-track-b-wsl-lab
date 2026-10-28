"""Keep this PC awake only while the authorized training pipeline is active."""
import ctypes,json,time,datetime,os
from pathlib import Path
root=Path('D:/mms_experiments/gg49-lanesegnet-lab_sj');status=root/'runs/pipeline-status.json'
flags=0x80000000|0x00000001
if not ctypes.windll.kernel32.SetThreadExecutionState(flags):raise OSError('Cannot establish process-scoped system-awake request')
(root/'runs/keep-awake.json').write_text(json.dumps({'pid':os.getpid(),'system_awake':True,'display_required':False,'global_power_settings_modified':False,'started_at':datetime.datetime.now().isoformat()}),encoding='utf-8')
failed_since=None
try:
    while True:
        try:d=json.loads(status.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):d={}
        if d.get('status')=='completed' and d.get('phase')=='done':break
        if d.get('status') in ['failed','blocked']:
            if failed_since is None:failed_since=time.time()
            if time.time()-failed_since>7200:break
        else:failed_since=None
        time.sleep(30)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    (root/'runs/keep-awake.json').write_text(json.dumps({'pid':os.getpid(),'system_awake':False,'ended_at':datetime.datetime.now().isoformat()}),encoding='utf-8')
