"""Short real GPU acceptance through the same HTTP interface used by the UI."""
import json, time, urllib.request, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,assets
base='http://127.0.0.1:'+str(__import__('os').getenv('VTC_PORT','8100'))
def api(path,payload=None):
    request=urllib.request.Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=240) as r: return json.load(r)
rows=[]
try:
    # Three deliveries plus genuine task outputs; every run keeps the same 32-frame default.
    cases=[('scanqa','raw','text'),('scanqa','numbered','text'),('scanqa','numbered','vtc')]
    cases += [(k,'numbered','vtc') for k in ('sqa3d','scan2cap','scanrefer','multi3drefer')]
    for key,mode,delivery in cases:
        payload={'dataset':key,'index':0,'input_mode':mode,'description_mode':delivery,'max_new_tokens':128,'enable_thinking':False}
        job_id=api('/api/infer',payload)['job_id']
        while True:
            job=api('/api/jobs/'+job_id)
            if job['status'] in {'done','error'}: break
            time.sleep(3)
        if job['status']=='error': raise RuntimeError(json.dumps(job))
        r=job['result']
        row={k:r.get(k) for k in ('dataset','run_id','input_mode','description_mode','input_tokens','output_tokens','prediction','parse_error','metrics','latency_s','peak_gpu_allocated_GiB','artifact_dir')}
        rows.append(row);print(json.dumps(row),flush=True)
        assets.write(Path(config.WEBUI_ROOT)/'results/gpu_acceptance_progress.json',{'status':'running','rows':rows})
    assets.write(Path(config.WEBUI_ROOT)/'results/gpu_acceptance.json',{'status':'completed','rows':rows,'not_accuracy_gate':True})
finally:
    api('/api/model/unload',{})
