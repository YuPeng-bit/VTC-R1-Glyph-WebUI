"""FastAPI backend for the VTC-R1-Glyph WebUI."""

from __future__ import annotations

import io
import re
from pathlib import Path
import json
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, engine as engine_mod, metrics as metrics_mod, store, assets
from .prompts import version_extra_prompt, version_output_prompt, PROMPT_VERSIONS

WEB_DIR = os.path.join(config.WEBUI_ROOT, "web")
RUNS_LOG = os.path.join(config.RESULTS_DIR, "runs.jsonl")

app = FastAPI(title="VTC-R1-Glyph WebUI", version="1.0.0")

# --------------------------------------------------------------------------
# job registry (single GPU -> one job at a time)
# --------------------------------------------------------------------------
class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self._queue = None
        self._worker = None
        self._start_worker()

    def _start_worker(self) -> None:
        import queue

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="vtc-worker", daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                continue
            job = self._jobs.get(job_id)
            if job is None:
                continue
            payload = job.pop("_payload")
            self._update(job_id, status="running", stage="inference", started_at=time.time())
            try:
                if not engine_mod.ENGINE.loaded:
                    self._update(job_id, stage="loading_model")
                self._update(job_id, stage="inference")
                if payload.get('record_only'):
                    engine_mod.ENGINE.ensure_loaded()
                    self._update(job_id,status='done',stage='done',result={'model_loaded':True})
                    continue
                result = engine_mod.ENGINE.run(**payload, run_id=job_id)
                result['gt'] = store.STORE.gt_display(payload['key'], store.STORE.get(payload['key'], payload['index']))
                if result.get('parse_error'):
                    result['metrics'] = {'error':result['parse_error'], '_status':'unscorable'}
                else:
                    try:
                        result['metrics'] = metrics_mod.METRICS.evaluate(payload['key'],store.STORE.get(payload['key'],payload['index']),result['metrics_input'])
                    except Exception as exc:
                        result['metrics'] = {'error':f'{type(exc).__name__}: {exc}', '_status':'evaluation_failed'}
                result["run_id"] = job_id
                result["finished_at"] = time.time()
                assets.write(os.path.join(result['artifact_dir'],'result.json'), result)
                self._append_run(result)
                self._update(job_id, status="done", stage="done", result=result)
                touch_activity()
            except Exception as exc:  # pragma: no cover
                folder=Path(config.RESULTS_DIR)/job_id
                folder.mkdir(parents=True,exist_ok=True)
                assets.write(folder/'failure.json',{'error':f'{type(exc).__name__}: {exc}',
                    'traceback':traceback.format_exc(),'dataset':payload.get('key'),'index':payload.get('index'),'time':time.time()})
                self._update(
                    job_id,
                    status="error",
                    stage="error",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            finally:
                if engine_mod.ENGINE.loaded:
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    def _append_run(self, result: dict) -> None:
        with open(RUNS_LOG, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(result, ensure_ascii=False, default=str)+'\n')
            fh.flush()

    def _update(self, job_id: str, **kw) -> None:
        with self._lock:
            self._jobs.setdefault(job_id, {}).update(kw)

    def submit(self, payload: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "submitted_at": time.time(),
                "dataset": payload.get('key'),
                "index": payload.get('index'),
                "_payload": payload,
            }
            self._order.append(job_id)
            if len(self._order) > 200:
                stale = self._order.pop(0)
                self._jobs.pop(stale, None)
        touch_activity()
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if not k.startswith("_")}

    def history(self, limit: int = 50) -> List[dict]:
        with self._lock:
            out = []
            for job_id in reversed(self._order[-limit:]):
                job = self._jobs.get(job_id)
                if not job:
                    continue
                item = {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "stage": job.get("stage"),
                    "submitted_at": job.get("submitted_at"),
                    "dataset": job.get('dataset'),
                    "index": job.get('index'),
                }
                res = job.get("result")
                if res:
                    item.update(
                        {
                            "dataset": res.get("dataset"),
                            "index": res.get("index"),
                            "id": res.get("id"),
                            "scene": res.get("scene"),
                            "prediction": res.get("prediction"),
                            "metrics": res.get("metrics"),
                            "input_mode": res.get('input_mode'),
                            "description_mode": res.get('description_mode'),
                        }
                    )
                out.append(item)
            return out

    def busy(self) -> bool:
        with self._lock:
            return any(j.get("status") in ("queued", "running") for j in self._jobs.values())


JOBS = JobRegistry()

# --------------------------------------------------------------------------
# idle auto-unload: keeps the shared lab GPU free when nobody is using the UI.
# 0 (or negative) disables it.  Override with V3D_IDLE_UNLOAD_SEC.
# --------------------------------------------------------------------------
IDLE_UNLOAD_SEC = int(os.environ.get("VTC_IDLE_UNLOAD_SEC", "600"))
LAST_ACTIVITY = time.time()
_ACTIVITY_LOCK = threading.RLock()


def touch_activity() -> None:
    global LAST_ACTIVITY
    with _ACTIVITY_LOCK:
        LAST_ACTIVITY = time.time()


def _idle_unload_loop() -> None:
    while True:
        time.sleep(30)
        if IDLE_UNLOAD_SEC <= 0:
            continue
        with _ACTIVITY_LOCK:
            idle = time.time() - LAST_ACTIVITY
        if idle < IDLE_UNLOAD_SEC:
            continue
        if not engine_mod.ENGINE.loaded or JOBS.busy():
            continue
        engine_mod.ENGINE.unload()
        touch_activity()


threading.Thread(target=_idle_unload_loop, name="v3d-idle-unload", daemon=True).start()


# --------------------------------------------------------------------------
# static frontend
# --------------------------------------------------------------------------
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>VTC-R1-Glyph WebUI</h1><p>web/index.html is missing.</p>", status_code=500)
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# --------------------------------------------------------------------------
# datasets / questions
# --------------------------------------------------------------------------
@app.get("/api/datasets")
def api_datasets() -> dict:
    out = []
    for key in config.DATASET_ORDER:
        info = config.dataset_info(key)
        try:
            n = store.STORE.count(key)
            err = None
        except Exception as exc:
            n, err = 0, f"{type(exc).__name__}: {exc}"
        out.append(
            {
                "key": key,
                "label": info["label"],
                "split": info["split"],
                "task_type": info["task_type"],
                "mode": info["mode"],
                "metric_set": info["metric_set"],
                "expects_box_input": info["expects_box_input"],
                "question_file": info["question_file"],
                "official_eval_script": info["official_eval_script"],
                "official_eval_cmd": info["official_eval_cmd"],
                "reference_metrics": info["reference_metrics"],
                "reference_note": info["reference_note"],
                "n_questions": n,
                "error": err,
            }
        )
    return {
        "datasets": out,
        "defaults": {
            "extra_prompt": version_extra_prompt(config.DEFAULT_PROMPT_VERSION),
            "prompt_version": config.DEFAULT_PROMPT_VERSION,
            "prompt_versions": {version:{"extra_prompt":version_extra_prompt(version),"output_prompts":{key:{mode:version_output_prompt(key,mode,version) for mode in ('raw','numbered')} for key in config.DATASET_ORDER}} for version in PROMPT_VERSIONS},
            "enable_thinking": config.DEFAULT_ENABLE_THINKING,
            "output_prompts": {key: {mode: version_output_prompt(key, mode, config.DEFAULT_PROMPT_VERSION) for mode in ('raw', 'numbered')} for key in config.DATASET_ORDER},
            "frame_strategy": config.DEFAULT_FRAME_SAMPLING_STRATEGY,
            "max_frames": config.DEFAULT_MAX_FRAME_NUM,
            "decode_strategy": config.DEFAULT_DECODE_STRATEGY,
            "temperature": config.DEFAULT_TEMPERATURE,
            "num_beams": config.DEFAULT_NUM_BEAMS,
            "max_new_tokens": config.DEFAULT_MAX_NEW_TOKENS,
            "strategies": config.FRAME_SAMPLING_STRATEGIES,
            "cot_order": config.COT_ORDER,
            "cot_presets": {
                k: {"label": v["label"], "instruction": v["instruction"], "style": v["style"],
                    "split": bool(v.get("split")), "force_prefix": v.get("force_prefix")}
                for k, v in config.COT_PRESETS.items()
            },
            "ckpt": config.CKPT_PATH,
            "repo": config.REPO_ROOT,
        },
    }


@app.get("/api/datasets/{key}/questions")
def api_questions(
    key: str,
    offset: int = 0,
    limit: int = Query(50, le=500),
    q: str = "",
) -> dict:
    _require_dataset(key)
    if q.strip():
        items = store.STORE.search(key, q, limit=limit)
        return {"total": len(items), "offset": 0, "items": items, "mode": "search"}
    total = store.STORE.count(key)
    offset = max(0, min(offset, max(0, total - 1)))
    items = [store.STORE.summary(key, i) for i in range(offset, min(total, offset + limit))]
    return {"total": total, "offset": offset, "items": items, "mode": "page"}


@app.get("/api/datasets/{key}/questions/{index}")
def api_question(key: str, index: int) -> dict:
    _require_dataset(key)
    try:
        record = store.STORE.get(key, index)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    scene = record.get("video")
    frames = store.STORE.scene_frames(scene)
    return {
        "key": key,
        "index": index,
        "id": record.get("id"),
        "scene": scene,
        "question_type": (record.get("metadata") or {}).get("question_type"),
        "conversation": record.get("conversations"),
        "prompt": (record.get("conversations") or [{}])[0].get("value", ""),
        "gt": store.STORE.gt_display(key, record),
        "box_input": record.get("box_input"),
        "box": record.get("box"),
        "annotations": record.get("annotations"),
        "metadata": record.get("metadata"),
        "n_frames": len(frames),
        "default_frame_indices": store.default_frame_indices(len(frames), config.DEFAULT_MAX_FRAME_NUM),
        "frames": store.STORE.frame_descriptors(scene),
    }


@app.get("/api/datasets/{key}/preview")
def api_preview(key: str, index: int, extra_prompt: str = "", question: str = "") -> dict:
    """Return exactly what would be sent to the model, without running it."""
    _require_dataset(key)
    plan = engine_mod.ENGINE.run(
        key,
        index,
        extra_prompt=extra_prompt if extra_prompt else None,
        question=question if question else None,
        record_only=True,
    )
    return plan


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------
@app.get("/api/media/frame")
def api_media_frame(scene: str, name: str, w: int = 0):
    if os.path.basename(scene) != scene or os.path.basename(name) != name:
        raise HTTPException(status_code=400, detail="invalid scene/name")
    if not name.lower().endswith((".jpg", ".png")):
        raise HTTPException(status_code=400, detail="only .jpg/.png are served")
    path = os.path.join(config.REPO_ROOT, "data", "scannet", "posed_images", scene, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="frame not found")
    if w and 0 < w <= 1024:
        return Response(content=_thumbnail(path, w), media_type="image/jpeg")
    return FileResponse(path, media_type="image/jpeg" if name.lower().endswith(".jpg") else "image/png")


def _thumbnail(path: str, w: int) -> bytes:
    from PIL import Image

    cache_name = f"{os.path.basename(os.path.dirname(path))}__{os.path.basename(path)}__w{w}.jpg"
    cache_path = os.path.join(config.THUMB_CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return fh.read()
    with Image.open(path) as img:
        img = img.convert("RGB")
        ratio = w / float(img.width)
        img = img.resize((w, max(1, int(round(img.height * ratio)))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
    data = buf.getvalue()
    try:
        with open(cache_path, "wb") as fh:
            fh.write(data)
    except Exception:
        pass
    return data


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
@app.get("/api/status")
def api_status() -> dict:
    eng = engine_mod.ENGINE
    gpu = _gpu_info()
    return {
        "model_loaded": eng.loaded,
        "model_loading": eng.loading,
        "load_error": eng.load_error,
        "load_seconds": eng.load_seconds,
        "model_name": eng.model_name,
        "ckpt_path": config.CKPT_PATH,
        "repo": config.REPO_ROOT,
        "device": str(eng.device) if eng.loaded else None,
        "selected_gpu": eng.selected_gpu if eng.loaded else None,
        "gpu_policy": 'highest free memory among visible GPUs before loading; fixed during inference',
        "gpu": gpu,
        "busy": JOBS.busy(),
        "idle_unload_sec": IDLE_UNLOAD_SEC,
        "idle_seconds": round(time.time() - LAST_ACTIVITY, 1),
    }


def _gpu_info() -> Optional[dict]:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]),
                     "memory_total_mb": int(parts[3]), "utilization": int(parts[4])}
                )
        return {"gpus": rows, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception:
        return None


@app.post("/api/model/load")
def api_model_load() -> dict:
    eng = engine_mod.ENGINE
    if eng.loaded:
        return {"model_loaded": True, "already": True, "load_seconds": eng.load_seconds}
    job_id = JOBS.submit({"key": "scanqa", "index": 0, "record_only": True})
    return {"job_id": job_id, "note": "the model is loaded lazily on the first inference request"}


@app.post("/api/model/unload")
def api_model_unload() -> dict:
    if JOBS.busy():
        raise HTTPException(status_code=409, detail="a job is running; try again when idle")
    return engine_mod.ENGINE.unload()


@app.post("/api/infer")
def api_infer(payload: Dict[str, Any] = Body(...)) -> dict:
    key = payload.get("dataset")
    _require_dataset(key)
    index = int(payload.get("index", 0))
    frame_indices = payload.get("frame_indices")
    if frame_indices is not None:
        frame_indices = [int(i) for i in frame_indices]
        if not frame_indices:
            raise HTTPException(status_code=400, detail="frame_indices is empty")
    run_payload = {
        "key": key,
        "index": index,
        "extra_prompt": payload.get("extra_prompt"),
        "prompt_version": payload.get("prompt_version",config.DEFAULT_PROMPT_VERSION),
        "output_prompt": payload.get("output_prompt"),
        "question": payload.get("question"),
        "frame_indices": frame_indices,
        "frame_strategy": payload.get("frame_strategy"),
        "max_frames": payload.get("max_frames"),
        "force_sample": payload.get("force_sample"),
        "decode_strategy": payload.get("decode_strategy"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "num_beams": payload.get("num_beams"),
        "max_new_tokens": payload.get("max_new_tokens"),
        "box_input": payload.get("box_input"),
        "enable_thinking": payload.get('enable_thinking',payload.get('cot_mode') != 'off' if payload.get('cot_mode') is not None else config.DEFAULT_ENABLE_THINKING),
        "input_mode": payload.get('input_mode','raw'),
        "description_mode": payload.get('description_mode','text'),
        "description_text": payload.get('description_text'),
        "relations_enabled": payload.get('relations_enabled',False),
        "relation_text": payload.get('relation_text'),
        "comparison_id": payload.get('comparison_id'),
        "image_max_pixels": payload.get('image_max_pixels',262144),
        "cot_mode": payload.get("cot_mode"),
        "cot_instruction": payload.get("cot_instruction"),
        "cot_force_prefix": payload.get("cot_force_prefix"),
        "cot_force_prefix_text": payload.get("cot_force_prefix_text"),
    }
    job_id = JOBS.submit(run_payload)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None and re.fullmatch(r'[a-f0-9]{12}',job_id):
        path=os.path.join(config.RESULTS_DIR,job_id,'result.json')
        failure=os.path.join(config.RESULTS_DIR,job_id,'failure.json')
        if os.path.isfile(path): job={'job_id':job_id,'status':'done','result':assets.read(path)}
        elif os.path.isfile(failure): job={'job_id':job_id,'status':'error',**assets.read(failure)}
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.get("/api/history")
def api_history(limit: int = Query(50, le=200)) -> dict:
    live=JOBS.history(limit=limit)
    seen={r['job_id'] for r in live}
    if os.path.isfile(RUNS_LOG):
        with open(RUNS_LOG,encoding='utf-8') as f:
            records=[]
            for line in f:
                try: records.append(json.loads(line))
                except ValueError: continue
        for r in reversed(records):
            if r['run_id'] in seen: continue
            live.append({'job_id':r['run_id'],'status':'done',**{k:r.get(k) for k in ('dataset','index','prediction','input_mode','description_mode')}})
    for path in sorted(Path(config.RESULTS_DIR).glob('*/failure.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        job_id=path.parent.name
        if job_id in seen: continue
        r=assets.read(path)
        live.append({'job_id':job_id,'status':'error','dataset':r.get('dataset'),'index':r.get('index'),'prediction':r.get('error')})
    return {'items':live[:limit]}


def _require_dataset(key: Optional[str]) -> None:
    if not key or key not in config.DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {key}")

@app.get('/api/scene-document')
def scene_document(scene: str):
    try:
        data=assets.scene_assets(scene)
        return {k:data[k] for k in ('scene','description_text','relation_text','relation_rule','inventory_path','inventory_sha256','caption_manifest','gt_fields_consumed')}
    except Exception as exc: raise HTTPException(400,detail=f'{type(exc).__name__}: {exc}')

@app.post('/api/preview')
def exact_preview(payload: dict=Body(...)):
    data=dict(payload); key=data.pop('dataset'); index=int(data.pop('index'))
    _require_dataset(key)
    try: return engine_mod.ENGINE.run(key,index,record_only=True,**data)
    except Exception as exc: raise HTTPException(400,detail=f'{type(exc).__name__}: {exc}')

@app.get('/api/media/asset')
def own_asset(path: str):
    root=Path(config.WEBUI_ROOT).resolve()/'results'
    target=(Path(config.WEBUI_ROOT)/path).resolve()
    if not target.is_relative_to(root) or not target.is_file(): raise HTTPException(404,detail='Unknown result asset')
    if target.suffix.lower() not in {'.png','.jpg','.html','.json','.txt'}: raise HTTPException(400,detail='Unsupported asset type')
    return FileResponse(target)
