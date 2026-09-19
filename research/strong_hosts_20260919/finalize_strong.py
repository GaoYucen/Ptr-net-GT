"""Finish this one bounded study after both training processes complete."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
scripts=Path(__file__).parent; launched={}; handles=[]
def run(script):
    subprocess.run([sys.executable,str(scripts/script),'--root',str(a.root)],check=True)
try:
    while True:
        for gpu,host in enumerate(('am','icam')):
            f=a.root/host/'status.json'
            if host not in launched and f.exists() and json.loads(f.read_text()).get('stage')=='complete':
                if (a.root/f'{host}-selector-traces.json').exists():
                    launched[host]=None
                else:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                    log=(a.root/f'{host}-selector-traces.log').open('w');handles.append(log)
                    launched[host]=subprocess.Popen([sys.executable,str(scripts/'trace_selector.py'),
                        '--root',str(a.root),'--host',host],env=env,stdout=log,stderr=subprocess.STDOUT)
                    print('START_TRACE',host,flush=True)
        for host,proc in launched.items():
            if proc is not None and proc.poll() not in (None,0):
                raise RuntimeError(f'{host} trace failed with {proc.returncode}')
        if len(launched)==2 and all(v is None or v.poll()==0 for v in launched.values()):break
        time.sleep(30)
    for host in ('am','icam'):
        for seed in (1234,4321,2468):
            for mode in ('route','random','learned'):
                assert (a.root/host/f'{mode}-seed{seed}/summary.json').exists()
    run('summarize_strong.py');run('make_report.py')
    for name in ('PROTOCOL.md','NEXT_STEPS.md'):
        (a.root/name).write_text((scripts/name).read_text())
    (a.root/'finalization.json').write_text(json.dumps(dict(stage='complete',arms=18,
        traces_complete=True,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
    run('export_evidence.py')
    print('STUDY_COMPLETE',flush=True)
except Exception as exc:
    (a.root/'finalization.json').write_text(json.dumps(dict(stage='failed',error=repr(exc)),indent=2))
    raise
finally:
    for f in handles:f.close()
