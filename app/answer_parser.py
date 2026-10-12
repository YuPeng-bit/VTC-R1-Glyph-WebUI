"""WebUI-only tolerant formatting parser. No GT, answer shortening or inference."""
from __future__ import annotations
import re

CONTROL_TOKENS = ('<|user|>', '<|assistant|>', '<|system|>', '<|observation|>',
                  '<|endoftext|>', '<eop>', '<sop>', '[gMASK]', '[sMASK]', '[MASK]')


def _boundary_controls(text):
    value = text.strip()
    removed = []
    while True:
        changed = False
        for token in CONTROL_TOKENS:
            if value.startswith(token):
                value = value[len(token):].strip(); removed.append(token); changed = True
            if value.endswith(token):
                value = value[:-len(token)].strip(); removed.append(token); changed = True
        if not changed:
            return value, removed


def parse_webui_answer(raw, completion=None, *, enable_thinking=False,
                       output_limit_reached=False, ended_with_eos=False):
    if not isinstance(raw, str):
        raise ValueError('Raw generation must be text')
    value, controls = _boundary_controls(raw)
    thinking = []
    # The native chat template can open thinking before generated tokens start.
    implicit = not value.startswith('<answer>') and (
        enable_thinking or ('</think>' in value and not value.startswith('<think>')))
    while value.startswith('<think>') or implicit:
        explicit = value.startswith('<think>')
        content = value[len('<think>'):] if explicit else value
        end = content.find('</think>')
        if end < 0:
            raise ValueError('Unfinished thinking; no final answer boundary')
        reason = content[:end]
        if '<think' in reason:
            raise ValueError('Nested or malformed thinking block')
        thinking.append(reason.strip())
        value = content[end + len('</think>'):].strip()
        implicit = False
        # Empty redundant terminators immediately after the reasoning boundary.
        while value.startswith('</think>'):
            value = value[len('</think>'):].strip()
    # A stray closing marker at the end must not hide the answer before it.
    redundant = 0
    while True:
        value, more = _boundary_controls(value)
        controls.extend(more)
        if not value.endswith('</think>'):
            break
        value = value[:-len('</think>')].strip()
        redundant += 1
    if '<think' in value or '</think' in value:
        raise ValueError('Unexpected thinking tag inside final answer')
    # Repair only this exact observed typo: one opening answer marker, answer
    # text, then a stray thinking closer and a genuine termination boundary.
    # Budget-truncated or ambiguously malformed answer markup is not repaired.
    repaired_closer = False
    terminal = ended_with_eos or any(t in controls for t in ('<|endoftext|>', '<|user|>', '<eop>'))
    if (redundant and terminal and value.startswith('<answer>')
            and value.count('<answer>') == 1 and '</answer' not in value
            and (not output_limit_reached or ended_with_eos)):
        value += '</answer>'
        repaired_closer = True
    tagged = '<answer' in value or '</answer' in value
    if tagged:
        match = re.fullmatch(r'<answer>([\s\S]*?)</answer>', value)
        if not match or value.count('<answer>') != 1 or value.count('</answer>') != 1:
            raise ValueError('Malformed or repeated answer tags, or text outside answer tags')
        answer = match.group(1).strip()
    else:
        if output_limit_reached and not ended_with_eos:
            raise ValueError('Output limit reached before a complete final answer; raw output preserved')
        answer = value.strip()
    if not answer:
        raise ValueError('No answer text after thinking')
    if any(token in answer for token in CONTROL_TOKENS):
        raise ValueError('Model control token occurs inside final answer')
    return answer, {
        'version': 'webui_answer_v3',
        'rule': 'paired_answer_tags' if tagged else ('plain_answer_after_completed_thinking' if thinking else 'plain_answer'),
        'official_check_completion': bool(completion(raw)) if completion else None,
        'thinking': '\n\n'.join(thinking) if thinking else None,
        'removed_boundary_controls': controls,
        'redundant_trailing_think_closers': redundant,
        'repaired_mismatched_answer_closer': repaired_closer,
        'ended_with_eos': ended_with_eos,
        'word_truncation': False, 'GT_consumed': False,
    }
