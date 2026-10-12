"""Bounded official corpus scoring; never average per-question BLEU/CIDEr."""
import json
import os
import signal
import subprocess
import sys

from . import config, metrics


def score_scanqa(predictions):
    records = {r['id']: r for r in metrics.METRICS._questions('scanqa')}
    gts, res = {}, {}
    correct = 0
    for sid_raw, prediction in predictions.items():
        sid = metrics.scanqa_normalize_id(sid_raw)
        if sid in gts:
            raise ValueError('ScanQA ID normalization would collapse predictions')
        refs = records[sid_raw]['metadata']['answers']
        gts[sid] = refs
        res[sid] = [str(prediction).rstrip('.')]
        correct += int(str(prediction) in refs)
    if not gts:
        raise ValueError('No predictions to score')
    timeout = int(os.environ.get('VTC_BATCH_SCORE_TIMEOUT_SEC', '300'))
    process = subprocess.Popen([sys.executable, '-m', 'app.corpus_worker'], cwd=config.WEBUI_ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(json.dumps({'gts': gts, 'res': res}), timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise RuntimeError(f'Official corpus scorer exceeded {timeout}s; predictions retained') from None
    if process.returncode:
        raise RuntimeError('Official corpus scorer failed: ' + stderr[-1500:])
    return {**json.loads(stdout), 'EM': correct / len(gts), 'count': len(gts)}
