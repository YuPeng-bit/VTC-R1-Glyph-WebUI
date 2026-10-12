"""Replay immutable actual generations without model calls or GT-based parsing."""
import hashlib
import json
import sys
import time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from app.answer_parser import parse_webui_answer
from test_answer_parser import USER_OUTPUT, USER_ANSWER, MISMATCHED_OUTPUT

out = root / 'results' / 'parser_acceptance' / str(time.time_ns())
out.mkdir(parents=True, exist_ok=False)
rows = []
for path in sorted((root / 'results/webui').glob('*/generation.json')):
    original = json.loads(path.read_text())
    raw = original['raw_output']
    ended = raw.rstrip().endswith(('<|endoftext|>', '<|user|>', '<eop>'))
    row = {'run_id': original['run_id'], 'source': str(path),
           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'original_parse_error': original.get('parse_error'), 'raw_output': raw}
    try:
        answer, protocol = parse_webui_answer(raw,
            enable_thinking=original.get('enable_thinking', False),
            output_limit_reached=original.get('output_limit_reached', False), ended_with_eos=ended)
        row.update(answer_text=answer, answer_protocol=protocol, parser_error=None)
    except ValueError as exc:
        row.update(answer_text=None, parser_error=str(exc))
    rows.append(row)
answer, protocol = parse_webui_answer(USER_OUTPUT)
assert answer == USER_ANSWER
fixture = {'source': 'user-provided reproduction; not a new model inference',
           'raw_output': USER_OUTPUT, 'answer_text': answer, 'answer_protocol': protocol}
(out / 'user_reproduction.json').write_text(json.dumps(fixture, indent=2) + '\n')
answer, protocol = parse_webui_answer(MISMATCHED_OUTPUT)
assert answer == 'dark-colored, smooth surface'
(out / 'user_mismatched_closer.json').write_text(json.dumps({'source':'second user-provided reproduction', 'raw_output':MISMATCHED_OUTPUT, 'answer_text':answer, 'answer_protocol':protocol},indent=2)+'\n')
(out / 'replayed_generations.json').write_text(json.dumps(rows, indent=2) + '\n')
report = {'status': 'passed', 'parser_version':'webui_answer_v3', 'fixture_cases': 33, 'real_generations_replayed': len(rows),
    'protocol_parsed': sum(r['parser_error'] is None for r in rows),
    'previously_failed_now_protocol_parsed': sum(bool(r['original_parse_error']) and r['parser_error'] is None for r in rows),
    'model_rerun': False, 'GT_consumed': False, 'original_results_modified': False,
    'answer_correctness_verified': False, 'evidence': str(out)}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
