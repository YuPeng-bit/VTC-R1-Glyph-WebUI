"""Durable ScanQA batches, executed by the same worker as interactive inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
import traceback
import uuid

from . import config

TERMINAL = {'completed', 'completed_with_errors', 'cancelled', 'error', 'interrupted'}
OPTIONS = {'input_mode', 'description_mode', 'relations_enabled', 'image_max_pixels',
           'enable_thinking', 'prompt_version', 'extra_prompt', 'output_prompt',
           'frame_strategy', 'max_frames', 'temperature', 'top_p', 'num_beams', 'max_new_tokens'}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(name + ' must be an integer')
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(name + ' must be an integer') from None
    if isinstance(value, float) and value != number:
        raise ValueError(name + ' must be an integer')
    if not low <= number <= high:
        raise ValueError(f'{name} must be {low}..{high}')
    return number


def validate_options(raw):
    if not isinstance(raw, dict):
        raise ValueError('options must be an object')
    unsupported = set(raw) - OPTIONS
    if unsupported:
        raise ValueError('Unsupported batch options: ' + ', '.join(sorted(unsupported)) +
                         '; questions, manually selected frames and scene text cannot be shared across a batch')
    options = {'input_mode': 'raw', 'description_mode': 'text', 'relations_enabled': False,
               'image_max_pixels': 262144, 'enable_thinking': config.DEFAULT_ENABLE_THINKING,
               'prompt_version': config.DEFAULT_PROMPT_VERSION, 'extra_prompt': None,
               'output_prompt': None, 'frame_strategy': 'uniform', 'max_frames': 32,
               'temperature': 0.0, 'top_p': None, 'num_beams': 1,
               'max_new_tokens': config.DEFAULT_MAX_NEW_TOKENS}
    options.update(raw)
    for name, allowed in [('input_mode', {'raw', 'numbered'}), ('description_mode', {'text', 'vtc'}),
                          ('prompt_version', {'original', 'optimized'}), ('frame_strategy', {'uniform'})]:
        if options[name] not in allowed:
            raise ValueError('Invalid ' + name)
    for name in ('enable_thinking', 'relations_enabled'):
        if type(options[name]) is not bool:
            raise ValueError(name + ' must be boolean')
    for name in ('extra_prompt', 'output_prompt'):
        if options[name] is not None and not isinstance(options[name], str):
            raise ValueError(name + ' must be text or null')
    for name, low, high in [('max_frames', 1, 64), ('image_max_pixels', 65536, 1048576),
                            ('max_new_tokens', 1, 8192), ('num_beams', 1, 16)]:
        options[name] = integer(options[name], name, low, high)
    for name, low, high in [('temperature', 0, 5), ('top_p', 0.000001, 1)]:
        if options[name] is None and name == 'top_p':
            continue
        try:
            value = float(options[name])
        except (TypeError, ValueError):
            raise ValueError('Invalid ' + name) from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid ' + name)
        options[name] = value
    from .prompts import version_extra_prompt, version_output_prompt
    if options['extra_prompt'] is None:
        options['extra_prompt'] = version_extra_prompt(options['prompt_version'])
    if options['output_prompt'] is None:
        options['output_prompt'] = version_output_prompt('scanqa', options['input_mode'], options['prompt_version'])
    return options


class BatchManager:
    def __init__(self, root, data_store, engine, score, activity=lambda: None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store, self.engine, self.score, self.activity = data_store, engine, score, activity
        self.lock = threading.RLock()
        self.active = set()
        self.cancel = {}
        # A server restart cannot silently make an unfinished job successful.
        for path in self.root.glob('*/state.json'):
            state = read_json(path)
            if state['status'] in {'queued', 'running', 'scoring', 'stopping'}:
                state.update(status='interrupted', stage='interrupted', updated_at=time.time(),
                             error='Server restarted; resume the saved selection and settings explicitly')
                write_json(path, state)

    def folder(self, batch_id):
        if not re.fullmatch(r'[a-f0-9]{12}', batch_id):
            raise ValueError('Invalid batch ID')
        folder = self.root / batch_id
        if not (folder / 'request.json').is_file():
            raise FileNotFoundError('Unknown batch')
        return folder

    def preview(self, payload):
        unknown = set(payload) - {'dataset', 'selection', 'count', 'seed', 'options'}
        if unknown:
            raise ValueError('Unknown batch request fields: ' + ', '.join(sorted(unknown)))
        if payload.get('dataset', 'scanqa') != 'scanqa':
            raise ValueError('Batch evaluation currently supports ScanQA val only')
        if config.dataset_info('scanqa')['split'] != 'val':
            raise ValueError('This batch entry requires the labelled ScanQA validation split')
        options = validate_options(payload.get('options', {}))
        records = self.store.questions('scanqa')
        if not records:
            raise ValueError('Empty ScanQA validation set')
        ids, normalized = set(), set()
        for record in records:
            sid = record.get('id')
            canonical = '_'.join(str(sid).split('_')[:-1] + ['0'])
            if not sid or sid in ids or canonical in normalized:
                raise ValueError('Duplicate/missing ScanQA question IDs; cannot score without collapsing samples')
            ids.add(sid); normalized.add(canonical)
            answers = (record.get('metadata') or {}).get('answers')
            if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                raise ValueError('Missing/non-text ScanQA reference answers: ' + sid)
        selection = payload.get('selection', 'random')
        seed = integer(payload.get('seed', 42), 'seed', 0, 2**31 - 1)
        if selection == 'all':
            indices = list(range(len(records)))
        elif selection == 'random':
            count = integer(payload.get('count', 20), 'count', 1, len(records))
            indices = sorted(random.Random(seed).sample(range(len(records)), count))
        else:
            raise ValueError('selection must be random or all')
        return {'dataset': 'scanqa', 'split': 'val', 'selection': selection, 'seed': seed,
                'total_available': len(records), 'selected_count': len(indices), 'indices': indices,
                'sample_ids': [records[i]['id'] for i in indices], 'options': options,
                'checkpoint': config.CKPT_PATH, 'question_file': config.abs_question_file('scanqa'),
                'question_file_sha256': fingerprint(config.abs_question_file('scanqa')),
                'scoring': 'Official corpus scorer on the selected/attempted set; failed answers are empty predictions',
                'input_policy': 'Each question uses its own scene, uniform frames and unedited scene document; GT is never model input'}

    def create(self, payload):
        with self.lock:
            if self.active:
                raise RuntimeError('A batch is already queued/running; stop or finish it first')
            request = self.preview(payload)
            batch_id = uuid.uuid4().hex[:12]
            folder = self.root / batch_id
            folder.mkdir()
            request.update(batch_id=batch_id, created_at=time.time())
            write_json(folder / 'request.json', request)
            write_json(folder / 'selection.json', [{'index': i, 'id': sid} for i, sid in zip(request['indices'], request['sample_ids'])])
            (folder / 'attempts.jsonl').touch()
            self._set(batch_id, status='queued', stage='queued', selected=request['selected_count'],
                      attempted=0, succeeded=0, failed=0, latest=None, metrics=None,
                      metrics_status='not_started', error=None, created_at=request['created_at'])
            self.active.add(batch_id)
            self.cancel[batch_id] = threading.Event()
            return self.get(batch_id)

    def _set(self, batch_id, **fields):
        with self.lock:
            path = self.root / batch_id / 'state.json'
            state = read_json(path) if path.exists() else {'batch_id': batch_id}
            state.update(fields, updated_at=time.time())
            write_json(path, state)
            return state

    def get(self, batch_id):
        with self.lock:
            folder = self.folder(batch_id)
            state = read_json(folder / 'state.json')
            state['request'] = read_json(folder / 'request.json')
            state['artifact_dir'] = str(folder)
            state['report_url'] = self.url(folder / 'report.html')
            state['files'] = {name: self.url(folder / name) for name in
                              ('request.json', 'selection.json', 'attempts.jsonl', 'predictions.jsonl', 'metrics.json')
                              if (folder / name).is_file()}
            return state

    def url(self, path):
        from urllib.parse import quote
        return '/api/media/asset?path=' + quote(str(Path(path).relative_to(config.WEBUI_ROOT)))

    def history(self, limit=30):
        with self.lock:
            paths = sorted(self.root.glob('*/state.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            return [self.get(p.parent.name) for p in paths]

    def stop(self, batch_id):
        with self.lock:
            state = self.get(batch_id)
            if batch_id in self.active:
                self.cancel[batch_id].set()
                self._set(batch_id, status='stopping', stage='stop_after_current_question')
            return self.get(batch_id)

    def resume(self, batch_id):
        with self.lock:
            if self.active:
                raise RuntimeError('A batch is already queued/running')
            state = self.get(batch_id)
            if state['status'] not in TERMINAL or state['status'] == 'completed':
                raise ValueError('This batch is not resumable')
            request = state['request']
            if request['checkpoint'] != config.CKPT_PATH:
                raise ValueError('Checkpoint changed; create a new batch rather than mixing weights')
            if request['question_file_sha256'] != fingerprint(config.abs_question_file('scanqa')):
                raise ValueError('ScanQA source changed; cannot resume the same batch')
            self.active.add(batch_id)
            self.cancel[batch_id] = threading.Event()
            folder = self.folder(batch_id)
            snapshot = folder / ('previous_' + str(time.time_ns()))
            snapshot.mkdir()
            for name in ('state.json', 'metrics.json', 'predictions.jsonl', 'report.html'):
                if (folder / name).is_file():
                    (snapshot / name).write_bytes((folder / name).read_bytes())
            # Keep snapshots while removing stale current-summary pointers.
            for name in ('metrics.json', 'report.html'):
                if (folder / name).is_file():
                    (folder / name).unlink()
            return self._set(batch_id, status='queued', stage='queued', error=None)

    def _rows(self, batch_id):
        """Append-only history: a truncated last line from a crash is ignored."""
        rows = {}
        path = self.folder(batch_id) / 'attempts.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row['index']] = row
        return rows

    def _append(self, batch_id, row):
        with open(self.folder(batch_id) / 'attempts.jsonl', 'a', encoding='utf-8') as fh:
            # Prefix a newline so an incomplete last record remains isolated on resume.
            fh.write('\n' + json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def run(self, batch_id):
        folder = self.folder(batch_id)
        request = read_json(folder / 'request.json')
        rows = self._rows(batch_id)
        stop = self.cancel[batch_id]
        terminal, error = 'completed', None
        succeeded = sum(r['status'] == 'ok' for r in rows.values())
        self._set(batch_id, status='running', stage='loading_model', started_at=time.time(), metrics=None,
                  metrics_status='not_started', finished_at=None, attempted=len(rows), succeeded=succeeded,
                  failed=len(rows)-succeeded)
        try:
            pending = [i for i in request['indices'] if rows.get(i, {}).get('status') != 'ok']
            if pending and not stop.is_set():
                self.engine.ensure_loaded()  # Fail once, rather than repeating a load failure thousands of times.
            for index in pending:
                if stop.is_set():
                    break
                self.activity()
                self._set(batch_id, stage='inference', current_index=index)
                record = self.store.get('scanqa', index)
                run_id = uuid.uuid4().hex[:12]
                case = {'index': index, 'id': record['id'], 'scene': record['video'], 'run_id': run_id,
                        'gt': (record.get('metadata') or {})['answers'], 'prediction': '', 'status': 'error'}
                try:
                    result = self.engine.run('scanqa', index, run_id=run_id,
                                             generation_seed=(request['seed'] + index) % (2**31), **request['options'])
                    result.update(gt=case['gt'], batch_id=batch_id)
                    if result.get('parse_error'):
                        case.update(status='parse_error', error=result['parse_error'])
                    elif not isinstance(result.get('metrics_input'), str):
                        case.update(status='parse_error', error='No text answer parsed')
                    else:
                        case.update(status='ok', prediction=result['metrics_input'])
                    case.update(raw_output=result.get('raw_output'), answer_text=result.get('answer_text'),
                                input_tokens=result.get('input_tokens'), output_tokens=result.get('output_tokens'),
                                latency_s=result.get('latency_s'), artifacts_url=result.get('artifacts_url'),
                                output_limit_reached=result.get('output_limit_reached'))
                    result['batch_em'] = int(case['status'] == 'ok' and case['prediction'] in case['gt'])
                    write_json(Path(result['artifact_dir']) / 'result.json', result)
                except Exception as exc:
                    case.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                    failure = Path(config.RESULTS_DIR) / run_id
                    failure.mkdir(exist_ok=True)
                    write_json(failure / 'failure.json', case)
                    if 'out of memory' in str(exc).lower():
                        terminal, error = 'error', case['error']
                case['EM'] = int(case['status'] == 'ok' and case['prediction'] in case['gt'])
                case['finished_at'] = time.time()
                self._append(batch_id, case)
                rows[index] = case
                succeeded = sum(r['status'] == 'ok' for r in rows.values())
                self._set(batch_id, attempted=len(rows), succeeded=succeeded, failed=len(rows) - succeeded,
                          latest=case, em_over_attempted=sum(r['EM'] for r in rows.values()) / len(rows))
                self.activity()
                if terminal == 'error':
                    break
            if stop.is_set():
                terminal = 'cancelled'
        except Exception as exc:
            terminal, error = 'error', f'{type(exc).__name__}: {exc}'
            write_json(folder / ('failure_' + str(time.time_ns()) + '.json'),
                       {'error': error, 'traceback': traceback.format_exc()})
        try:
            ordered = [rows[i] for i in request['indices'] if i in rows]
            predictions = [{'sample_id': r['id'], 'pred_response': r['prediction'] if r['status'] == 'ok' else ''} for r in ordered]
            (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions), encoding='utf-8')
            if ordered:
                self._set(batch_id, stage='scoring', metrics_status='running')
                try:
                    metrics = self.score({r['id']: r['prediction'] if r['status'] == 'ok' else '' for r in ordered})
                    metrics.update(EM=sum(r['EM'] for r in ordered) / len(ordered),
                                   attempted=len(ordered), selected=request['selected_count'],
                                   failed=sum(r['status'] != 'ok' for r in ordered),
                                   scope='selected_set' if len(ordered) == request['selected_count'] else 'partial_attempted_set')
                    write_json(folder / 'metrics.json', metrics)
                    self._set(batch_id, metrics=metrics, metrics_status='completed')
                except Exception as exc:
                    terminal, error = 'error', 'Corpus scoring failed: ' + str(exc)
                    write_json(folder / ('scoring_failure_' + str(time.time_ns()) + '.json'),
                               {'error': error, 'traceback': traceback.format_exc()})
                    self._set(batch_id, metrics_status='error', metrics=None)
            if terminal == 'completed' and any(r['status'] != 'ok' for r in ordered):
                terminal = 'completed_with_errors'
            self._set(batch_id, status=terminal, stage=terminal, error=error, finished_at=time.time(),
                      current_index=None, full_selected_set_attempted=len(ordered) == request['selected_count'])
            self._report(batch_id, ordered)
        finally:
            with self.lock:
                self.active.discard(batch_id)
            self.activity()
        return self.get(batch_id)

    def _report(self, batch_id, rows):
        from html import escape
        state = self.get(batch_id)
        lines = ['<meta charset="utf-8"><title>ScanQA batch</title><h1>ScanQA val 批量评测</h1>',
                 '<p>状态：' + escape(state['status']) + '；已尝试 ' + str(state['attempted']) + '/' + str(state['selected']) +
                 '；失败 ' + str(state['failed']) + '。失败答案计为空串，不删除失败题。</p>',
                 '<pre>' + escape(json.dumps(state.get('metrics'), ensure_ascii=False, indent=2)) + '</pre>',
                 '<table border="1" cellpadding="8"><tr><th>索引/ID</th><th>状态</th><th>GT</th><th>预测</th><th>EM</th><th>记录</th></tr>']
        for row in rows:
            link = row.get('artifacts_url')
            result_link = self.url(Path(config.RESULTS_DIR) / row['run_id'] / ('result.json' if row['status'] != 'error' else 'failure.json'))
            links = ('<a href="' + escape(link, quote=True) + '">图片与 Prompt</a> · ') if link else ''
            links += '<a href="' + escape(result_link, quote=True) + '">输入/输出/GT</a>'
            lines.append('<tr><td>' + escape(str(row['index']) + ' / ' + row['id']) + '</td><td>' + escape(row['status']) +
                         '</td><td>' + escape(json.dumps(row['gt'], ensure_ascii=False)) + '</td><td>' + escape(row['prediction']) +
                         '</td><td>' + str(row['EM']) + '</td><td>' + links + '</td></tr>')
        lines.append('</table><h2>固定设置</h2><pre>' + escape(json.dumps(state['request'], ensure_ascii=False, indent=2)) + '</pre>')
        (self.folder(batch_id) / 'report.html').write_text(''.join(lines), encoding='utf-8')
