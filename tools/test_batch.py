"""CPU regressions with explicitly synthetic predictions; no GPU claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from app import batch, config


class FakeStore:
    def __init__(self):
        self.records = [{'id': f'scanqa_val-scene0000-{i}_0', 'video': f'scene{i:04d}_00',
                         'metadata': {'answers': ['one']}} for i in range(5)]

    def questions(self, key):
        return self.records

    def get(self, key, index):
        return self.records[index]


class FakeEngine:
    def __init__(self, root):
        self.root, self.calls, self.loads = root, [], 0
        self.fail = set()
        self.callback = None
        self.load_failure = False

    def ensure_loaded(self):
        self.loads += 1
        if self.load_failure:
            raise RuntimeError('No visible GPU has enough memory')

    def run(self, key, index, run_id, **kw):
        self.calls.append((index, copy.deepcopy(kw)))
        folder = self.root / run_id
        folder.mkdir()
        if self.callback:
            self.callback()
        if index in self.fail:
            return {'artifact_dir': str(folder), 'parse_error': 'synthetic malformed answer',
                    'raw_output': '<think>synthetic</think>', 'metrics_input': None}
        return {'artifact_dir': str(folder), 'parse_error': None, 'raw_output': '<answer>one</answer>',
                'metrics_input': 'one', 'answer_text': 'one', 'input_tokens': 10, 'output_tokens': 4,
                'artifacts_url': '/api/media/asset?path=results/test.html'}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = config.WEBUI_ROOT, config.RESULTS_DIR, config.CKPT_PATH, config.REPO_ROOT
        config.WEBUI_ROOT = str(self.root)
        config.RESULTS_DIR = str(self.root / 'results/webui')
        config.CKPT_PATH = 'synthetic-model'
        config.REPO_ROOT = str(self.root / 'upstream')
        self.data = FakeStore()
        path = Path(config.abs_question_file('scanqa'))
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.data.records), encoding='utf-8')
        self.case_root = Path(config.RESULTS_DIR)
        self.case_root.mkdir(parents=True)
        self.engine = FakeEngine(self.case_root)
        self.scored = []
        self.manager = batch.BatchManager(self.case_root / 'batches', self.data, self.engine, self.score)

    def score(self, predictions):
        self.scored.append(dict(predictions))
        return {'CIDEr': 0.0, 'count': len(predictions)}

    def tearDown(self):
        config.WEBUI_ROOT, config.RESULTS_DIR, config.CKPT_PATH, config.REPO_ROOT = self.old
        self.temp.cleanup()

    def create(self, **kw):
        return self.manager.create({'selection': 'all', **kw})['batch_id']

    def test_fixed_random_selection_and_all(self):
        payload = {'selection': 'random', 'count': 3, 'seed': 42}
        a = self.manager.preview(payload)
        self.assertEqual(a, self.manager.preview(payload))
        self.assertEqual(len(set(a['indices'])), 3)
        self.assertEqual(self.manager.preview({'selection': 'all'})['indices'], list(range(5)))

    def test_rejects_invalid_counts_and_scene_specific_overrides(self):
        for payload in [{'count': 6}, {'count': 1.2}, {'seed': -1}, {'dataset': 'sqa3d'},
                        {'options': {'question': 'Do not repeat this question'}},
                        {'options': {'frame_indices': [0]}}, {'options': {'description_text': 'Scene A only'}},
                        {'options': {'max_frames': 0}}, {'options': {'temperature': float('nan')}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.manager.preview(payload)

    def test_no_duplicate_or_missing_gt(self):
        self.data.records[1]['id'] = self.data.records[0]['id'][:-1] + '3'
        with self.assertRaises(ValueError):
            self.manager.preview({})
        self.data.records[1]['id'] = 'unique_0'
        self.data.records[1]['metadata']['answers'] = []
        with self.assertRaises(ValueError):
            self.manager.preview({})

    def test_error_answers_are_not_dropped_resume_retries_only_failed(self):
        self.engine.fail = {2}
        bid = self.create(options={'extra_prompt': '', 'output_prompt': '', 'max_frames': 7})
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'completed_with_errors')
        self.assertEqual((result['attempted'], result['succeeded'], result['failed']), (5, 4, 1))
        self.assertEqual(result['metrics']['EM'], .8)
        self.assertEqual(len(self.scored[-1]), 5)
        self.assertEqual(list(self.scored[-1].values()).count(''), 1)
        original = self.engine.calls[2][1]
        self.engine.fail.clear()
        self.manager.resume(bid)
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([i for i, _ in self.engine.calls], [0, 1, 2, 3, 4, 2])
        self.assertEqual(original, self.engine.calls[-1][1])
        self.assertEqual(original['extra_prompt'], '')
        self.assertNotIn('question', original)
        self.assertTrue(list(self.manager.folder(bid).glob('previous_*')))

    def test_stop_preserves_subset_and_continues_same_selection(self):
        bid = self.create()
        self.engine.callback = lambda: self.manager.stop(bid)
        result = self.manager.run(bid)
        self.assertEqual((result['status'], result['attempted']), ('cancelled', 1))
        self.assertEqual(result['metrics']['scope'], 'partial_attempted_set')
        self.engine.callback = None
        self.manager.resume(bid)
        result = self.manager.run(bid)
        self.assertEqual(result['attempted'], 5)
        self.assertEqual([i for i, _ in self.engine.calls], list(range(5)))

    def test_queued_stop_does_not_load_model(self):
        bid = self.create()
        self.manager.stop(bid)
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(self.engine.loads, 0)

    def test_restart_is_interrupted_not_done(self):
        bid = self.create()
        restarted = batch.BatchManager(self.case_root / 'batches', self.data, self.engine, self.score)
        self.assertEqual(restarted.get(bid)['status'], 'interrupted')
        restarted.resume(bid)
        self.assertEqual(restarted.run(bid)['status'], 'completed')

    def test_refuses_mixed_model_or_changed_dataset(self):
        bid = self.create()
        self.manager.stop(bid)
        self.manager.run(bid)
        config.CKPT_PATH = 'different-model'
        with self.assertRaises(ValueError):
            self.manager.resume(bid)
        config.CKPT_PATH = 'synthetic-model'
        Path(config.abs_question_file('scanqa')).write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.manager.resume(bid)

    def test_gpu_load_failure_stops_once_and_keeps_selection(self):
        self.engine.load_failure = True
        bid = self.create()
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['attempted'], 0)
        self.assertEqual(self.engine.loads, 1)
        self.assertEqual(result['selected'], 5)

    def test_scoring_failure_retry_does_not_rerun_inference(self):
        bid = self.create()
        self.manager.score = lambda _: (_ for _ in ()).throw(RuntimeError('synthetic METEOR timeout'))
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'error')
        self.assertTrue((self.manager.folder(bid) / 'predictions.jsonl').is_file())
        self.manager.score = self.score
        self.manager.resume(bid)
        result = self.manager.run(bid)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.engine.calls), 5)


if __name__ == '__main__':
    unittest.main()
