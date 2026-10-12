"""CPU-only genuine input checks for editable output prompts/native defaults."""
import json
import sys
import time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from app import config, assets
from app.engine import ENGINE
from app.prompts import default_output_prompt

out=root/'results/prompt_acceptance'/str(time.time_ns())
out.mkdir(parents=True,exist_ok=False)
processor=ENGINE.ensure_processor()
fixture=[{'role':'user','content':[{'type':'text','text':'Template check'}]}]
official_default=processor.apply_chat_template(fixture,tokenize=False,add_generation_prompt=True)
explicit_native=processor.apply_chat_template(fixture,tokenize=False,add_generation_prompt=True,enable_thinking=True)
assert official_default==explicit_native
assert '/nothink' not in explicit_native
off=processor.apply_chat_template(fixture,tokenize=False,add_generation_prompt=True,enable_thinking=False)
assert '/nothink' in off
rows=[]
for key in config.DATASET_ORDER:
    for mode in ('raw','numbered'):
        plan=ENGINE.run(key,0,record_only=True,input_mode=mode,prompt_version='original')
        assert plan['enable_thinking'] is True and plan['max_new_tokens']==8192
        assert '/nothink' not in plan['chat_template']
        assert plan['model_prompt'].endswith(default_output_prompt(key,mode))
        assert plan['output_prompt']==default_output_prompt(key,mode)
        assert not plan['output_prompt_manually_edited']
        assert plan['provenance']['gt_fields_consumed']==[]
        assert len(plan['frame_names'])==32
        assets.write(out/f'{key}_{mode}.json',plan)
        rows.append({'task':key,'mode':mode,'tokens':plan['input_tokens']})
        print(json.dumps(rows[-1]),flush=True)
for name,kw in [('custom',{'output_prompt':'Return exactly <answer>YOUR_RESPONSE</answer>.'}),
                ('empty',{'output_prompt':''}),('off',{'enable_thinking':False,'max_new_tokens':512})]:
    plan=ENGINE.run('scanqa',0,record_only=True,prompt_version='original',**kw)
    if name=='custom':
        assert plan['model_prompt'].endswith(kw['output_prompt'])
        assert plan['output_prompt']==kw['output_prompt'] and plan['output_prompt_manually_edited']
        assert default_output_prompt('scanqa') not in plan['model_prompt']
    elif name=='empty':
        assert plan['output_prompt']=='' and default_output_prompt('scanqa') not in plan['model_prompt']
    else:
        assert not plan['enable_thinking'] and '/nothink' in plan['chat_template']
        assert plan['max_new_tokens']==512
    assets.write(out/f'override_{name}.json',plan)
report={'status':'passed','actual_input_plans':len(rows)+3,'rows':rows,
    'default_matches_original_processor_template':True,'default_max_new_tokens':8192,
    'custom_prompt_replaces_default':True,'empty_prompt_supported':True,
    'thinking_off_supported':True,'model_loaded':ENGINE.loaded,'GPU_generation_run':False,'evidence':str(out)}
assert not ENGINE.loaded
assets.write(out/'report.json',report)
print(json.dumps(report),flush=True)
