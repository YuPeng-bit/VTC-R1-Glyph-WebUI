"""Formatting regressions, including the exact user-reported output. CPU only."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer_parser import parse_webui_answer

USER_OUTPUT = "<think>Got it, let's see. The question is asking about the type of chairs around the table. From the images, I can see that most of the chairs have a patterned fabric, likely a checkered or plaid design. Some chairs are wooden with a more classic design, and a few are black. The patterned chairs are the most common. So the answer should mention the patterned chairs as the primary type.</think>patterned chairs with checkered design and some wooden chairs and black chairs.</think><|endoftext|>"
USER_ANSWER = 'patterned chairs with checkered design and some wooden chairs and black chairs.'
MISMATCHED_OUTPUT = "<think>Got it, let's see. The question is asking about the top of the table. From the images, the table has a dark-colored top, probably a dark brown or black. The chairs around it are light brown with dark legs. The table is rectangular and has a smooth surface. So the top of the table is a dark-colored, smooth surface.</think><answer>dark-colored, smooth surface</think><|user|>"

class AnswerParserTests(unittest.TestCase):
    def test_mismatched_closer_requires_real_termination(self):
        answer, info = parse_webui_answer(MISMATCHED_OUTPUT)
        self.assertEqual(answer, 'dark-colored, smooth surface')
        self.assertTrue(info['repaired_mismatched_answer_closer'])
        for raw, kw in [
            ('</think><answer>dark</think>', {}),
            ('</think><answer>dark<|user|>', {}),
            ('</think><answer>dark</think><|user|>', {'output_limit_reached':True}),
            ('</think><answer>dark<answer>light</think><|user|>', {}),
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_webui_answer(raw, **kw)

    def test_user_output_preserves_all_answer_words(self):
        answer, info = parse_webui_answer(USER_OUTPUT)
        self.assertEqual(answer, USER_ANSWER)
        self.assertEqual(info['redundant_trailing_think_closers'], 1)
        self.assertFalse(info['GT_consumed'])

    def test_supported_formats(self):
        cases = [
            ('<think>reason</think>brown chairs<|endoftext|>', 'brown chairs', {}),
            ('reason</think>brown chairs', 'brown chairs', {'enable_thinking': True}),
            ('</think>brown chairs</think></think><|user|>', 'brown chairs', {}),
            ('<think>reason</think></think>brown chairs', 'brown chairs', {}),
            ('brown wooden chairs<|endoftext|>', 'brown wooden chairs', {}),
            ('<answer>brown chairs</answer><|endoftext|>', 'brown chairs', {}),
            ('<answer>brown chairs</answer></think><|endoftext|>', 'brown chairs', {}),
            ('<answer>brown chairs</answer>', 'brown chairs', {'enable_thinking': True}),
            ('<think>reason</think><answer>brown chairs</answer></think>', 'brown chairs', {}),
            ('<think><answer>not final</answer></think><think>again</think>green', 'green', {}),
            ('<|assistant|>reason</think>[7, 12]<|user|>', '[7, 12]', {}),
            ('</think><answer>7</answer>', '7', {'output_limit_reached': True}),
            ('</think>one<|endoftext|>', 'one', {'output_limit_reached': True, 'ended_with_eos': True}),
        ]
        for raw, expected, kwargs in cases:
            with self.subTest(raw=raw):
                self.assertEqual(parse_webui_answer(raw, **kwargs)[0], expected)

    def test_reject_ambiguous_or_unfinished_outputs(self):
        cases = [
            ('<think>unfinished', {}), ('<think>only reason</think>', {}),
            ('reason without boundary', {'enable_thinking': True}),
            ('</think>unfinished answer', {'output_limit_reached': True}),
            ('unfinished text', {'output_limit_reached': True}),
            ('</think><answer>brown', {}), ('</think></answer>brown<answer>', {}),
            ('<answer>brown</answer>extra', {}),
            ('<answer>brown</answer><answer>green</answer>', {}),
            ('<answer> </answer>', {}), ('</think>brown <|user|> chair', {}),
            ('<think>nested <think>x</think>answer', {}),
            ('</think>answer<think>unfinished', {}),
            ('</think>answer</think>extra', {}),
        ]
        for raw, kwargs in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_webui_answer(raw, **kwargs)

if __name__ == '__main__':
    unittest.main()
