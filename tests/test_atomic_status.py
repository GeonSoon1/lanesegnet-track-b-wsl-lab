"""Regressions for the observed transient SSD status-file rename failure."""
import errno,json
from concurrent.futures import ThreadPoolExecutor
from gg49_lab import common

def test_disappearing_temporary_file_is_retried_without_partial_target(tmp_path,monkeypatch):
    target=tmp_path/'status.json'
    common.write_json(target,{'step':10})
    replace=common.os.replace;attempts=[]
    def transient(source,dest):
        attempts.append(source)
        if len(attempts)==1:
            source.unlink()
            assert json.loads(target.read_text())=={'step':10}
            raise FileNotFoundError(errno.ENOENT,'Simulated observed missing temporary file')
        return replace(source,dest)
    monkeypatch.setattr(common.os,'replace',transient)
    common.write_json(target,{'step':11})
    assert json.loads(target.read_text())=={'step':11}
    assert len(attempts)==2 and attempts[0]!=attempts[1]
    assert list(tmp_path.glob('*.tmp'))==[]

def test_threads_in_one_process_never_share_a_temporary_name(tmp_path):
    target=tmp_path/'status.json'
    def write(step):common.write_json(target,{'step':step,'payload':'x'*100})
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(write,range(40)))
    assert json.loads(target.read_text())['step'] in range(40)
    assert list(tmp_path.glob('*.tmp'))==[]
