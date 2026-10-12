"""Editable task output instructions; independent of questions and GT."""
FORMAT_INSTRUCTION = (
    'Required final-answer format: <answer>YOUR_ANSWER</answer>. '
    'Replace YOUR_ANSWER with your actual answer; do not copy the placeholder. '
    'If you reason, finish the reasoning with </think> before opening <answer>. '
    'Close the final answer with </answer>, never with </think>. '
    'Write exactly one final-answer block and nothing after </answer>. '
)


def default_output_prompt(task, mode='raw'):
    if task in {'scanqa', 'sqa3d'}:
        style = 'Use a concise English answer phrase. For counts, use English number words. Syntax example only: <answer>blue</answer>.'
    elif task == 'scan2cap':
        style = 'Describe the target object in English. Put only the description inside the answer block.'
    elif mode == 'numbered' and task == 'scanrefer':
        style = 'Put only one object number inside the answer block, for example <answer>7</answer>. Do not output a bounding box, JSON object, or explanation inside the answer block.'
    elif mode == 'numbered' and task == 'multi3drefer':
        style = 'Put a list of all matching object numbers inside the answer block, for example <answer>[7, 12]</answer>, or <answer>[]</answer> when none match. Do not output bounding boxes, a JSON object, or explanations inside the answer block.'
    else:
        style = 'Explain in English which object or objects the description refers to. These RGB frames supply no explicit object IDs or world-coordinate reference for 3D boxes.'
    return FORMAT_INSTRUCTION + style
