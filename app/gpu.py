"""Select one sufficiently free visible GPU before loading; no process eviction."""
import os, subprocess

def query_gpus():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    rows=[]
    for line in result.stdout.splitlines():
        index,uuid,free,total=[part.strip() for part in line.split(',')]
        rows.append({'physical_index':int(index),'uuid':uuid,'free_GiB':float(free)/1024,'total_GiB':float(total)/1024})
    return rows

def select_gpu(rows, visible, minimum):
    # Default launcher exposes GPUs by UUID, avoiding CUDA/NVML ordinal ambiguity.
    requested=[x.strip() for x in visible.split(',')] if visible is not None else [r['uuid'] for r in rows]
    candidates=[]
    for logical_id,item in enumerate(requested):
        matches=[r for r in rows if item==str(r['physical_index']) or item and r['uuid'].startswith(item)]
        if len(matches)!=1: continue
        candidates.append({**matches[0],'logical_index':logical_id})
    suitable=[r for r in candidates if r['free_GiB']>=minimum]
    if not suitable:
        detail=', '.join(f"GPU{r['physical_index']}: {r['free_GiB']:.1f} GiB free" for r in candidates) or 'no visible GPU'
        raise RuntimeError(f'No visible GPU has the required {minimum:.1f} GiB free ({detail}). No other process will be stopped.')
    return max(suitable,key=lambda r:(r['free_GiB'],-r['logical_index']))
