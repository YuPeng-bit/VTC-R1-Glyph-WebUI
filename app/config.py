"""Native VTC configuration; official dataset registry retained from reference WebUI."""
import os
from pathlib import Path
REPO_ROOT=os.environ.get('V3D_REPO',str(Path(__file__).resolve().parents[2]/'Video-3D-LLM'))
WEBUI_ROOT=os.environ.get('VTC_WEBUI',str(Path(__file__).resolve().parents[1]))
CKPT_PATH=os.environ.get('VTC_CKPT','yiboowang/VTC-R1-Glyph')
RESULTS_DIR=os.path.join(WEBUI_ROOT,'results','webui')
THUMB_CACHE_DIR=os.path.join(WEBUI_ROOT,'results','thumbcache')
LOG_DIR=os.path.join(WEBUI_ROOT,'logs')
for folder in (RESULTS_DIR,THUMB_CACHE_DIR,LOG_DIR): os.makedirs(folder,exist_ok=True)
DEFAULT_EXTRA_PROMPT='The images are RGB frames from the same indoor scene. Use the visible evidence to answer the question.\n'
DEFAULT_FRAME_SAMPLING_STRATEGY='uniform'
DEFAULT_MAX_FRAME_NUM=32
DEFAULT_FORCE_SAMPLE=True
DEFAULT_TEMPERATURE=0.0
DEFAULT_TOP_P=None
DEFAULT_NUM_BEAMS=1
DEFAULT_MAX_NEW_TOKENS=8192
DEFAULT_ENABLE_THINKING=True
DEFAULT_PROMPT_VERSION='optimized'
DEFAULT_DECODE_STRATEGY='native_glm_processor'
FRAME_SAMPLING_STRATEGIES=['uniform']
COT_PRESETS={'off':{'label':'原生思考关闭','instruction':'','style':'native','split':False},'native':{'label':'原生思考开启','instruction':'','style':'native','split':True}}
COT_ORDER=['native','off']
DATASETS = {
    "scanqa": {
        "key": "scanqa",
        "label": "ScanQA",
        "dataset_field": "scanQA",
        "question_file": "data/processed/scanqa_val_llava_style.json",
        "split": "val",
        "mode": "generate",
        "metric_set": "scanqa",
        "task_type": "question answering",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_scanqa.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scanqa.sh $CKPT uniform 32",
        "reference_metrics": {
            "CIDEr": 102.16229907535768,
            "BLEU-1": 47.101587072802296,
            "BLEU-2": 31.622544537462165,
            "BLEU-3": 22.593420680668856,
            "BLEU-4": 15.81619851121378,
            "METEOR": 19.876435066133315,
            "ROUGE-L": 49.076598864059285,
            "EM": 0.30117647058823527,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scanqa_baseline.txt",
    },
    "scan2cap": {
        "key": "scan2cap",
        "label": "Scan2Cap",
        "dataset_field": "scan2cap",
        "question_file": "data/processed/scan2cap_val_llava_style.json",
        "split": "val",
        "mode": "caption",
        "metric_set": "scan2cap",
        "task_type": "dense captioning",
        "expects_box_input": True,
        "official_eval_script": "llava/eval/eval_scan2cap.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scan2cap.sh $CKPT uniform 32",
        "reference_metrics": {
            "CIDEr": 83.83649561980529,
            "BLEU-4": 41.29272209370019,
            "METEOR": 28.821606148812823,
            "ROUGE-L": 62.305666961213355,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scan2cap_baseline.txt",
    },
    "scanrefer": {
        "key": "scanrefer",
        "label": "ScanRefer",
        "dataset_field": "scanrefer",
        "question_file": "data/processed/scanrefer_vg_val_llava_style.json",
        "split": "val",
        "mode": "grounding",
        "metric_set": "scanrefer",
        "task_type": "3D visual grounding",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_scanrefer.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_scanrefer.sh $CKPT uniform 32",
        "reference_metrics": {
            "IoU@0.25": 58.182583087925956,
            "IoU@0.5": 51.75641564997896,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/scanrefer_baseline.txt",
    },
    "multi3drefer": {
        "key": "multi3drefer",
        "label": "Multi3DRefer",
        "dataset_field": "multi3drefer",
        "question_file": "data/processed/multi3drefer_val_llava_style.json",
        "split": "val",
        "mode": "grounding",
        "metric_set": "multi3drefer",
        "task_type": "multi-object 3D visual grounding",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_multi3drefer.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_multi3drefer.sh $CKPT uniform 32",
        "reference_metrics": {
            "F1@0.25": 0.574089126896188,
            "F1@0.5": 0.5213496196772585,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/multi3drefer_baseline.txt (0-1 scale, threshold 0.4)",
    },
    "sqa3d": {
        "key": "sqa3d",
        "label": "SQA3D (test)",
        "dataset_field": "sqa3d",
        "question_file": "data/processed/sqa3d_test_llava_style.json",
        "split": "test",
        "mode": "generate",
        "metric_set": "sqa3d",
        "task_type": "situated question answering",
        "expects_box_input": False,
        "official_eval_script": "llava/eval/eval_sqa3d.py",
        "official_eval_cmd": "sh scripts/3d/eval/eval_sqa3d.sh $CKPT uniform 32",
        "reference_metrics": {
            "EM(acc)": 58.48252344416027,
        },
        "reference_note": "reproduced baseline (uniform, 32 frames) recorded in results/gt_structure_full/evaluation/sqa3d_baseline.txt",
    },
}

DATASET_ORDER = ["scanqa", "scan2cap", "scanrefer", "multi3drefer", "sqa3d"]


def dataset_info(key: str) -> dict:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset: {key}. known: {list(DATASETS)}")
    return DATASETS[key]


def abs_question_file(key: str) -> str:
    return os.path.join(REPO_ROOT, dataset_info(key)["question_file"])
