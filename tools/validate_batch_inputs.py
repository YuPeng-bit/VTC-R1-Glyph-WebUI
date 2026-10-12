"""Real CPU input and upstream scoring checks. Does not generate model answers."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from app import config, engine, store
from app.batch import BatchManager, write_json
from app.batch_scoring import score_scanqa


def main():
    root = Path(config.WEBUI_ROOT) / 'results/batch_checks' / str(time.time_ns())
    root.mkdir(parents=True)
    manager = BatchManager(root / 'batches', store.STORE, engine.ENGINE, score_scanqa)
    selected = manager.preview({'selection': 'random', 'count': 3, 'seed': 42})
    assert selected == manager.preview({'selection': 'random', 'count': 3, 'seed': 42})
    full = manager.preview({'selection': 'all'})
    assert len(full['indices']) == store.STORE.count('scanqa')
    write_json(root / 'random_selection.json', selected)
    write_json(root / 'full_selection.json', full)
    plans = []
    index = selected['indices'][0]
    for mode, delivery in [('raw', 'text'), ('numbered', 'text'), ('numbered', 'vtc')]:
        options = dict(selected['options'], input_mode=mode, description_mode=delivery,
                       relations_enabled=mode == 'numbered')
        plan = engine.ENGINE.run('scanqa', index, record_only=True, **options)
        assert plan['provenance']['gt_fields_consumed'] == []
        assert plan['frame_strategy'] == 'uniform'
        assert plan['frame_indices'] == store.default_frame_indices(plan['n_frames_available'], 32)
        assert plan['input_tokens'] + plan['max_new_tokens'] <= 32768
        name = mode + '_' + delivery
        write_json(root / (name + '_input.json'), plan)
        (root / (name + '_prompt.txt')).write_text(plan['model_prompt'], encoding='utf-8')
        plans.append({'mode': mode, 'delivery': delivery, 'index': index,
                      'input_tokens': plan['input_tokens'], 'frames': len(plan['frame_indices']),
                      'images': len(plan['actual_images'])})
    # Fixed synthetic hypotheses test the scorer, not the VTC model. References
    # are supplied only to scoring; there is no GPU/model answer generation.
    records = store.STORE.questions('scanqa')
    predictions = {r['id']: ('chair' if i % 2 else 'one') for i, r in enumerate(records[:10])}
    ours = score_scanqa(predictions)
    path = root / 'scorer_fixture_predictions.jsonl'
    path.write_text(''.join(json.dumps({'sample_id': sid, 'pred_response': pred}) + '\n'
                            for sid, pred in predictions.items()), encoding='utf-8')
    # Execute the actual upstream script, bypassing model imports only.
    code = '''import sys, types, runpy
from pathlib import Path
repo=Path(sys.argv[1])
for name,path in [('llava',repo/'llava'),('llava.eval',repo/'llava/eval')]:
 p=types.ModuleType(name); p.__path__=[str(path)]; sys.modules[name]=p
script=repo/'llava/eval/eval_scanqa.py'
sys.argv=[str(script),'--input-file',sys.argv[2]]
runpy.run_path(str(script),run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code, config.REPO_ROOT, str(path)],
                            cwd=config.REPO_ROOT, text=True, capture_output=True, timeout=300)
    (root / 'official_script_stdout.txt').write_text(result.stdout, encoding='utf-8')
    (root / 'official_script_stderr.txt').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    official = {}
    for label, key in [('CIDER', 'CIDEr'), ('METEOR', 'METEOR'), ('Rouge', 'ROUGE-L'), ('EM', 'EM')]:
        official[key] = float(re.search(r'^' + label + r':\s*(\S+)', result.stdout, re.M)[1])
    bleu = re.search(r'^BLEU:\s*(.+)', result.stdout, re.M)[1].split(',')
    official.update({f'BLEU-{i}': float(value) for i, value in enumerate(bleu, 1)})
    differences = {key: abs(ours[key] - value) for key, value in official.items()}
    assert max(differences.values()) < 1e-8, differences
    report = {'status': 'passed', 'available_questions': full['selected_count'], 'random_questions': 3,
              'genuine_cpu_input_plans': plans, 'official_script_metric_differences': differences,
              'scorer_fixture_count': len(predictions), 'fixture_is_synthetic_not_model_output': True,
              'gpu_generation_tested': False, 'root': str(root)}
    write_json(root / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
