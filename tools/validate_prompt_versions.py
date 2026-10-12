"""Real CPU inputs: version preservation, exact user text, overrides, no GPU."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from app import assets,config
from app.engine import ENGINE
from app.prompts import (version_output_prompt,version_question,ORIGINAL_EXTRA_PROMPT,
                         OPTIMIZED_EXTRA_PROMPT,OPTIMIZED_OUTPUT_PROMPT)
out=root/'results/prompt_versions'/str(time.time_ns())
out.mkdir(parents=True,exist_ok=False)
old_source = (root/'tools/fixtures/original_prompts_20261010.py').read_text()
namespace={}
exec(compile(old_source,'preserved-original-prompts','exec'),namespace)
for task in config.DATASET_ORDER:
    for mode in ('raw','numbered'):
        assert version_output_prompt(task,mode,'original')==namespace['default_output_prompt'](task,mode)
assert ORIGINAL_EXTRA_PROMPT==config.DEFAULT_EXTRA_PROMPT
assert version_question('What color? Answer the question simply.','optimized')=='What color? Answer with one word or phrase.'
plans=[]
for name,kwargs in [
    ('default',{}), ('original',{'prompt_version':'original'}),
    ('explicit_new',{'prompt_version':'optimized'}),
    ('custom',{'prompt_version':'optimized','extra_prompt':'CUSTOM_CONTEXT','question':'CUSTOM_QUESTION','output_prompt':'CUSTOM_FORMAT'}),
    ('empty',{'prompt_version':'optimized','extra_prompt':'','output_prompt':''}),
]:
    plan=ENGINE.run('scanqa',0,record_only=True,**kwargs)
    assert plan['enable_thinking'] is True and plan['max_new_tokens']==8192
    assert plan['provenance']['gt_fields_consumed']==[]
    if name in ('default','explicit_new'):
        assert plan['prompt_version']=='optimized'
        assert plan['extra_prompt']==OPTIMIZED_EXTRA_PROMPT
        assert plan['output_prompt']==OPTIMIZED_OUTPUT_PROMPT
        assert 'Answer with one word or phrase' in plan['question']
        assert 'Answer the question simply' not in plan['question']
        assert not plan['output_prompt_manually_edited']
    elif name=='original':
        assert plan['extra_prompt']==ORIGINAL_EXTRA_PROMPT
        assert plan['output_prompt']==namespace['default_output_prompt']('scanqa','raw')
        assert 'Answer the question simply' in plan['question']
    elif name=='custom':
        assert plan['model_prompt']=='CUSTOM_CONTEXT\nCUSTOM_QUESTION\n\nCUSTOM_FORMAT'
        assert plan['extra_prompt_manually_edited'] and plan['question_manually_edited'] and plan['output_prompt_manually_edited']
    else:
        assert plan['extra_prompt']=='' and plan['output_prompt']==''
        assert OPTIMIZED_OUTPUT_PROMPT not in plan['model_prompt']
    assets.write(out/f'{name}.json',plan)
    plans.append(plan)
assert plans[0]['model_prompt']==plans[2]['model_prompt']
assert all(p['frame_names']==plans[0]['frame_names'] and p['actual_images']==plans[0]['actual_images'] for p in plans)
assert not ENGINE.loaded
report={'status':'passed','actual_inputs':len(plans),'default_version':'optimized',
    'original_prompt_source':'tools/fixtures/original_prompts_20261010.py','original_prompt_source_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
    'original_task_mode_outputs_preserved':10,'exact_user_text_preserved':True,
    'question_rewrite_only_requested_phrase':True,'manual_and_empty_overrides_pass':True,
    'same_32_rgb_frames':True,'GPU_generation_run':False,'evidence':str(out)}
assets.write(out/'report.json',report)
print(json.dumps(report),flush=True)
