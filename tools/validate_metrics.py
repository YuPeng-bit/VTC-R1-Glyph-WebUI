"""Scorer sanity checks on public GT self-matches. Not model predictions/results."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, assets,store
from app.metrics import METRICS,clean_answer
rows=[]
for key in config.DATASET_ORDER:
    rec=store.STORE.get(key,0);gt=store.STORE.gt_display(key,rec)
    prediction=gt if key in {'scanrefer','multi3drefer'} else gt[0]
    score=METRICS.evaluate(key,rec,prediction)
    if key=='scanqa': assert score['EM']==1
    if key=='sqa3d': assert score['EM(acc)']==1
    if key=='scanrefer': assert score['IoU']>.999
    if key=='multi3drefer': assert score['F1@0.25']>.999 and score['F1@0.5']>.999
    assert 'error' not in score
    rows.append({'dataset':key,'scores':score}); print(json.dumps(rows[-1]),flush=True)
assert clean_answer('1')==clean_answer('one')
assets.write(Path(config.WEBUI_ROOT)/'results/scorer_acceptance.json',{'status':'passed','test_type':'GT self-match scorer sanity; no model inference or effectiveness claim','rows':rows})
