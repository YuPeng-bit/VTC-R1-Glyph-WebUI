"""Finite CPU acceptance on genuine public examples; records actual model inputs."""
import json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, assets, store
from app.engine import ENGINE, parse_prediction

out=Path(config.WEBUI_ROOT)/'results/acceptance'/time.strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True,exist_ok=False)
rows=[]
for key in config.DATASET_ORDER:
    baseline=None
    for mode,delivery,relations in [('raw','text',False),('numbered','text',False),('numbered','vtc',True)]:
        started=time.time()
        plan=ENGINE.run(key,0,record_only=True,input_mode=mode,description_mode=delivery,relations_enabled=relations,max_new_tokens=128)
        assets.write(out/f'{key}_{mode}_{delivery}.json',plan)
        if baseline is None: baseline=plan['frame_names']
        assert plan['frame_names']==baseline
        assert len(plan['frame_names'])==32
        assert all(message['role']=='user' for message in plan['messages'])
        assert plan['provenance']['gt_fields_consumed']==[]
        if mode=='raw':
            assert not plan['document'] and not plan['numbered_frames'] and not plan['render_manifest']
        else:
            inv=assets.scene_assets(plan['scene'])['objects']
            valid_ids={str(o['display_id']) for o in inv}
            assert all(label['display_id'] in valid_ids for frame in plan['numbered_frames'] for label in frame['labels'])
            assert 'T1:' not in plan['document'] and 'T2:' not in plan['document'] and 'T3:' not in plan['document']
        rows.append({'dataset':key,'mode':mode,'delivery':delivery,'relations':relations,'frames':len(plan['frame_names']),
            'pages':len(plan['render_manifest']['pages']) if plan['render_manifest'] else 0,'input_tokens':plan['input_tokens'],
            'seconds':time.time()-started})
        print(json.dumps(rows[-1]),flush=True)

# Counterfactual GT mutation must not change any model input (caption target is legitimate task input).
for key in config.DATASET_ORDER:
    rec=store.STORE.get(key,0)
    original=json.loads(json.dumps(rec))
    kw={'record_only':True,'input_mode':'numbered','description_mode':'text','max_new_tokens':128}
    before=ENGINE.run(key,0,**kw)
    for turn in rec['conversations']:
        if turn.get('from') in {'gpt','assistant'}: turn['value']='GT_SENTINEL_NOT_MODEL_INPUT'
    rec.setdefault('metadata',{})['answers']=['GT_SENTINEL_NOT_MODEL_INPUT']
    rec['annotations']=['GT_SENTINEL_NOT_MODEL_INPUT'];rec['box']=['GT_SENTINEL_NOT_MODEL_INPUT'];rec['gt_box']=['GT_SENTINEL_NOT_MODEL_INPUT']
    after=ENGINE.run(key,0,**kw)
    assert before['messages']==after['messages'] and before['input_tokens']==after['input_tokens']
    rec.clear();rec.update(original)

objects=[{'display_id':'7','center_m':[1.,2.,3.],'extent_m':[1.,1.,1.]}]
assert parse_prediction('scanrefer','7','numbered',objects)[1]==[1.,2.,3.,1.,1.,1.]
assert parse_prediction('multi3drefer','[]','numbered',objects)[1]==[]
for key,text in [('scanrefer','I think 7'),('scanrefer','99'),('multi3drefer','not sure'),('multi3drefer','[7,7]')]:
    assert parse_prediction(key,text,'numbered',objects)[2]
assert parse_prediction('scanrefer','7','raw',objects)[2]
report={'status':'passed','checks':'15 genuine uniform32 inputs, shared frame identity, native processor token lengths, object label alignment, GT counterfactual invariance for five tasks, strict grounding parsers',
        'gpu_inference_tested':False,'rows':rows,'max_input_tokens':max(r['input_tokens'] for r in rows),'directory':str(out)}
assets.write(out/'report.json',report)
assets.write(Path(config.WEBUI_ROOT)/'results/acceptance/latest.json',report)
print(json.dumps({'report':str(out/'report.json'),'status':'passed'}),flush=True)
