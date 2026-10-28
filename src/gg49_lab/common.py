from pathlib import Path
import json,os,time,datetime,uuid,errno
ROOT=Path(__file__).resolve().parents[2]
STORAGE=json.loads((ROOT/'configs/storage.local.json').read_text())
STORE=Path(STORAGE['storage_root'])
RAW=Path(STORAGE['raw_root']);SHP=Path(STORAGE['shp_root'])
PYTHON=str(Path(STORAGE['environment'])/'bin/python')
# Keep module invocations consistent with the shell launcher as well.
if os.path.ismount('/mnt/d'):
    for key,relative in {'TMPDIR':'cache/tmp','TMP':'cache/tmp','TEMP':'cache/tmp','MPLCONFIGDIR':'cache/matplotlib','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','PIP_CACHE_DIR':'cache/pip','CUDA_CACHE_PATH':'cache/cuda'}.items():
        os.environ[key]=str(STORE/relative)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_json(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {} if default is None else default
def ensure_storage():
    if not os.path.ismount("/mnt/d"):
        raise RuntimeError("D: is unavailable; no fallback to C: is permitted")
def write_json(p,obj):
    ensure_storage()
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    for attempt in range(4):
        tmp=p.with_name(p.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        try:
            with tmp.open('x',encoding='utf-8') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
            return
        except OSError as e:
            if e.errno not in [errno.ENOENT,errno.EACCES,errno.EBUSY] or attempt==3:raise
            time.sleep(.05*(2**attempt))
        finally:
            try:tmp.unlink(missing_ok=True)
            except OSError:pass
def status(phase,state='running',message='',**extra):
    p=STORE/'runs/pipeline-status.json';d=read_json(p)
    completed=d.get('completed_phases',[])
    if d.get('phase') and d.get('phase')!=phase and d.get('status')=='running':completed=list(dict.fromkeys(completed+[d['phase']]))
    d.update(phase=phase,status=state,message=message,updated_at=now(),pid=os.getpid(),completed_phases=completed,**extra);write_json(p,d)
    print(json.dumps({'phase':phase,'status':state,'message':message},ensure_ascii=False),flush=True)
def append_jsonl(p,obj):
    ensure_storage()
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),**obj),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
