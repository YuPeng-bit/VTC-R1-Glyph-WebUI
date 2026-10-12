"""Native, unmodified VTC-R1-Glyph inference. No Video3D connector or fine-tuned weights."""
from __future__ import annotations
import gc, json, os, re, threading, time, uuid
from pathlib import Path
import numpy as np
from . import config, store, assets
from .gpu import query_gpus, select_gpu
from .answer_parser import parse_webui_answer
from .prompts import validate_prompt_version, version_extra_prompt, version_output_prompt, version_question
from native_answer_protocol import pinned_completion

def parse_prediction(key, text, mode, objects):
    if key not in {'scanrefer', 'multi3drefer'}: return text, text, None
    if mode != 'numbered':
        return None, None, 'Raw RGB supplies neither explicit object IDs nor world geometry; 3D grounding is not scored in this mode.'
    value = text.strip()
    if key == 'scanrefer':
        match = re.fullmatch(r'(?:Object\s+)?(\d+)', value, re.I)
        if not match: return None, None, 'Expected a single object ID; no ID is inferred from reasoning.'
        ids = [match.group(1)]
    else:
        if not re.fullmatch(r'\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]', value):
            return None, None, 'Expected a list of object IDs such as [7, 12] or []; malformed output is not treated as an empty prediction.'
        ids = [str(i) for i in json.loads(value)]
        if len(ids) != len(set(ids)): return None, None, 'Repeated object IDs'
    lookup = {str(o['display_id']): o for o in objects}
    if any(i not in lookup for i in ids): return None, None, 'Unknown object ID'
    boxes = [lookup[i]['center_m']+lookup[i]['extent_m'] for i in ids]
    return ids[0] if key == 'scanrefer' else ids, boxes[0] if key == 'scanrefer' else boxes, None

class NativeEngine:
    def __init__(self):
        self.lock = threading.RLock(); self.model = None; self.processor = None
        self.loading = False; self.load_error = None; self.load_seconds = None
        self.device = 'cuda:0'; self.model_name = 'VTC-R1-Glyph (native)'
        self.selected_gpu = None
        self.completion = pinned_completion(assets.PROJECT/'src/t50/vendor/vtc_r1_inference.py')

    @property
    def loaded(self): return self.model is not None

    def ensure_processor(self):
        with self.lock:
            if self.processor is None:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(config.CKPT_PATH, local_files_only=True)
            return self.processor

    def ensure_loaded(self):
        with self.lock:
            if self.loaded: return
            self.loading = True; started = time.time()
            try:
                import torch
                if not torch.cuda.is_available(): raise RuntimeError('No visible CUDA GPU. Preview remains usable without loading the model.')
                minimum=float(os.getenv('VTC_MIN_FREE_GIB','26'))
                chosen=select_gpu(query_gpus(),os.getenv('CUDA_VISIBLE_DEVICES'),minimum)
                self.device=f"cuda:{chosen['logical_index']}"
                torch.cuda.set_device(chosen['logical_index'])
                free,_=torch.cuda.mem_get_info(chosen['logical_index'])
                if free < int(minimum*2**30):
                    raise RuntimeError(f"GPU{chosen['physical_index']} free memory changed to {free/2**30:.1f} GiB; needs {minimum:.1f} GiB. Retry after memory is available.")
                self.selected_gpu={**chosen,'free_GiB_before_load':free/2**30}
                from transformers import Glm4vForConditionalGeneration
                self.ensure_processor()
                model = Glm4vForConditionalGeneration.from_pretrained(config.CKPT_PATH, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa')
                model.eval(); model.to(self.device); self.model = model
                self.load_seconds = time.time()-started; self.load_error = None
            except Exception as exc:
                self.load_error = str(exc); self.model = None; gc.collect()
                raise
            finally: self.loading = False

    def unload(self):
        with self.lock:
            self.model = None; gc.collect()
            import torch
            if torch.cuda.is_initialized() and self.selected_gpu is not None:
                with torch.cuda.device(self.device): torch.cuda.empty_cache()
            self.selected_gpu = None
        return {'model_loaded': False}

    def plan(self, key, index, **kw):
        record = store.STORE.get(key, index)
        scene = record['video']; frames = store.STORE.scene_frames(scene)
        if scene not in store.STORE.frame_index(): raise ValueError('Scene absent from official frame metadata; no filesystem fallback allowed')
        selected = kw.get('frame_indices')
        if selected is None: selected = store.default_frame_indices(len(frames), int(kw.get('max_frames') or 32))
        if not selected or len(selected) != len(set(selected)) or any(type(i) is not int or i < 0 or i >= len(frames) for i in selected):
            raise ValueError('Select unique valid frame indices')
        if len(selected) > 64: raise ValueError('At most 64 frames per interactive run')
        selected = sorted(selected)
        mode = kw.get('input_mode') or 'raw'; delivery = kw.get('description_mode') or 'text'
        if mode not in {'raw','numbered'} or delivery not in {'text','vtc'}: raise ValueError('Unknown input mode')
        if kw.get('frame_strategy') not in {None,'uniform'}: raise ValueError('Only the reference uniform selection is supported')
        prompt_version = validate_prompt_version(kw.get('prompt_version') or config.DEFAULT_PROMPT_VERSION)
        original_question = re.sub(r'<image>|<ground>|<coord>', '', record['conversations'][0]['value']).strip()
        question = kw.get('question')
        if question is None: question = version_question(original_question, prompt_version)
        question = re.sub(r'<image>|<ground>|<coord>', '', question).strip()
        context = version_extra_prompt(prompt_version) if kw.get('extra_prompt') is None else kw['extra_prompt']
        document = ''; scene_data = None; view_records = []
        images = [store.to_abs(frames[i]) for i in selected]
        prompt = context+'\n'+question
        target = None
        if key == 'scan2cap':
            center = kw.get('box_input') if kw.get('box_input') is not None else record.get('box_input')
            if center is None or len(center) < 3: raise ValueError('Scan2Cap requires its task-provided target center')
            center = [float(x) for x in center[:3]]
            if not np.isfinite(center).all(): raise ValueError('Invalid target center')
            prompt += '\nTask-provided target center in axis-aligned world meters: '+str([round(x,2) for x in center])
            target = {'center_m': center, 'source': 'task input, not caption GT', 'mapping': None}
        if mode == 'numbered':
            scene_data = assets.scene_assets(scene)
            descriptions = kw.get('description_text')
            if descriptions is None: descriptions = scene_data['description_text']
            if not descriptions.strip(): raise ValueError('Numbered mode needs nonempty object descriptions')
            document = 'Scene objects. The same object number identifies the same predicted object in every photograph and in this document. Some objects may be occluded or absent from a selected frame; missing labels do not prove absence.\n\n'+descriptions
            if kw.get('relations_enabled'):
                relations = kw.get('relation_text')
                if relations is None: relations = scene_data['relation_text']
                document += '\n\n'+relations
            view_records = [assets.numbered_frame(scene, path) for path in images]
            images = [v['path'] for v in view_records]
            prompt += '\nUse the numbered scene photographs to locate candidate objects and match their numbers to the object document. Check both sources before answering. The document consists of model-generated descriptions and may contain errors.'
            if target:
                nearest = min(scene_data['objects'], key=lambda o: np.linalg.norm(np.asarray(o['center_m'])-target['center_m']))
                target['mapping'] = {'display_id': nearest['display_id'], 'rule':'nearest predicted center, no GT overlap lookup',
                    'distance_m': float(np.linalg.norm(np.asarray(nearest['center_m'])-target['center_m']))}
                prompt += f"\nDescribe predicted object {nearest['display_id']}, selected by nearest center to the task-provided target."
            if delivery == 'text': prompt += '\n\nObject document:\n'+document
            else: prompt += '\nThe object document is in the text-rendered pages following the scene photographs. Read those pages to look up the matching numbers.'
        output_prompt = kw.get('output_prompt')
        if output_prompt is None: output_prompt = version_output_prompt(key, mode, prompt_version)
        if not isinstance(output_prompt, str): raise ValueError('output_prompt must be text')
        if output_prompt: prompt += '\n\n'+output_prompt
        pages = assets.glyph_pages(document) if mode == 'numbered' and delivery == 'vtc' else None
        ordered = [{'kind': 'scene_rgb', 'path': p, 'sha256': assets.file_sha256(p),
                    'url': view_records[j]['url'] if view_records else store.STORE.frame_descriptors(scene)[selected[j]]['rgb']}
                   for j,p in enumerate(images)]
        if pages: ordered.extend({'kind':'vtc_page', **p} for p in pages['pages'])
        thinking = bool(kw.get('enable_thinking', config.DEFAULT_ENABLE_THINKING))
        processor = self.ensure_processor()
        pixels = int(kw.get('image_max_pixels') or 262144)
        if pixels < 65536 or pixels > 1048576: raise ValueError('image_max_pixels must be 65536..1048576')
        ordered = [assets.processor_input(row,pixels) for row in ordered]
        content=[]
        for row in ordered:
            title='Rendered object-description page:' if row['kind']=='vtc_page' else 'Scene RGB photograph:'
            content.extend([{'type':'text','text':title},{'type':'image','image':row['path']}])
        content.append({'type':'text','text':prompt})
        messages=[{'role':'user','content':content}]
        template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        from PIL import Image
        pil_images=[Image.open(row['path']).convert('RGB') for row in ordered]
        try:
            encoded=processor(text=[template],images=pil_images,return_tensors='pt',
                size={'shortest_edge':12544,'longest_edge':pixels})
        finally:
            for image in pil_images: image.close()
        max_new = int(kw.get('max_new_tokens') or config.DEFAULT_MAX_NEW_TOKENS)
        if not 1 <= max_new <= 8192: raise ValueError('max_new_tokens must be 1..8192')
        tokens = int(encoded['input_ids'].shape[-1])
        if tokens+max_new > 32768: raise ValueError(f'Context exceeds 32768: {tokens} input + {max_new} reserved output; reduce input explicitly')
        plan = {'dataset':key,'index':index,'id':record.get('id'),'scene':scene,'question':question,
            'original_question':original_question,'prompt_version':prompt_version,'extra_prompt':context,
            'extra_prompt_manually_edited':context != version_extra_prompt(prompt_version),
            'question_manually_edited':question != version_question(original_question, prompt_version),
            'input_mode':mode, 'description_mode':delivery if mode == 'numbered' else None,
            'relations_enabled': bool(mode == 'numbered' and kw.get('relations_enabled')),
            'frame_indices':selected,'frame_names':[Path(frames[i]).name for i in selected], 'n_frames_available':len(frames),
            'frame_strategy':'uniform' if kw.get('frame_indices') is None else 'manual', 'max_frames':kw.get('max_frames') or 32,
            'decode_strategy':'native_glm_processor', 'model_prompt':prompt, 'output_prompt':output_prompt,
            'output_prompt_manually_edited':output_prompt != version_output_prompt(key, mode, prompt_version),
            'messages':messages, 'document':document,
            'actual_images':ordered,'numbered_frames':view_records,'render_manifest':pages,
            'input_tokens':tokens,'image_max_pixels':pixels,'image_grid_thw':encoded.get('image_grid_thw').tolist() if encoded.get('image_grid_thw') is not None else None,
            'chat_template': template,
            'enable_thinking':thinking, 'max_new_tokens':max_new,'temperature':float(kw.get('temperature') or 0),
            'comparison_id':kw.get('comparison_id'),
            'num_beams':int(kw.get('num_beams') or 1), 'target_input':target,
            'provenance':{ 'weights':config.CKPT_PATH, 'native_model':True,'spatial_tokens':False,'gt_fields_consumed':[],
                'source_inventory_sha256':scene_data['inventory_sha256'] if scene_data else None,
                'caption_manifest_sha256':scene_data['caption_manifest_sha256'] if scene_data else None,
                'descriptions_manually_edited': bool(scene_data and kw.get('description_text') is not None and kw['description_text'] != scene_data['description_text']),
                'relation_rule':scene_data['relation_rule'] if scene_data and kw.get('relations_enabled') else None}}
        return plan, encoded, scene_data

    def run(self, key, index, record_only=False, run_id=None, **kw):
        with self.lock:
            if record_only: return self.plan(key,index,**kw)[0]
            run_id = run_id or uuid.uuid4().hex[:12]
            folder = Path(config.RESULTS_DIR)/run_id; folder.mkdir(parents=True,exist_ok=False)
            assets.write(folder/'request.json', {'key':key,'index':index,**kw})
            assets.write(folder/'evaluation_gt.json', {'gt':store.STORE.gt_display(key,store.STORE.get(key,index)),
                'use':'display and post-generation evaluation only; never model input'})
            try:
                plan, encoded, scene_data = self.plan(key,index,**kw)
            except Exception as exc:
                assets.write(folder/'failure.json',{'error':f'{type(exc).__name__}: {exc}','time':time.time()})
                raise
            assets.write(folder/'input.json', plan)
            (folder/'prompt.txt').write_text(plan['model_prompt'],encoding='utf-8')
            (folder/'document.txt').write_text(plan['document'],encoding='utf-8')
            from PIL import Image
            from html import escape
            # Exact RGB files supplied to native processor; native patch grid is also recorded.
            for j,row in enumerate(plan['actual_images']):
                os.symlink(row['path'], folder/f"input_{j:03d}{Path(row['path']).suffix}")
            (folder/'inputs.html').write_text('<meta charset="utf-8"><h1>Actual input images</h1>'+''.join(
                f'<h3>{j}: {escape(row["kind"])}</h3><img style="max-width:900px" src="{escape(row["url"])}">' for j,row in enumerate(plan['actual_images']))+'<h2>Actual prompt</h2><pre>'+escape(plan['model_prompt'])+'</pre>',encoding='utf-8')
            try:
                self.ensure_loaded()
                import torch
                started = time.time(); torch.cuda.reset_peak_memory_stats(self.device)
                batch = {k:(v.to(self.device) if hasattr(v,'to') else v) for k,v in encoded.items()}
                generation = {'max_new_tokens':plan['max_new_tokens'],'do_sample':plan['temperature']>0,'num_beams':plan['num_beams']}
                if generation['do_sample']: generation['temperature']=plan['temperature']
                if kw.get('top_p') is not None and generation['do_sample']: generation['top_p']=float(kw['top_p'])
                with torch.inference_mode(): output = self.model.generate(**batch, **generation)
                ids = output[0,batch['input_ids'].shape[-1]:].tolist()
                raw = self.processor.decode(ids,skip_special_tokens=False)
                parse_error = None; answer = None; thinking = None
                try:
                    eos_ids = self.model.generation_config.eos_token_id
                    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
                    answer, protocol = parse_webui_answer(raw, self.completion,
                        enable_thinking=plan['enable_thinking'],
                        output_limit_reached=len(ids)>=plan['max_new_tokens'],
                        ended_with_eos=bool(ids and ids[-1] in eos_ids))
                    thinking = protocol['thinking']
                    pred, metric_input, parse_error = parse_prediction(key,answer,plan['input_mode'],scene_data['objects'] if scene_data else [])
                except Exception as exc:
                    protocol = {'error':str(exc)}; pred = metric_input = None; parse_error = str(exc)
                result = {**plan,'run_id':run_id,'raw_output':raw,'generated_token_ids':ids,'output_tokens':len(ids),
                    'answer_text':answer,'prediction':pred,'metrics_input':metric_input,'parse_error':parse_error,
                    'thinking':thinking,'answer_protocol':protocol,'latency_s':time.time()-started,
                    'output_limit_reached':len(ids)>=plan['max_new_tokens'], 'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated(self.device)/2**30,
                    'selected_gpu':self.selected_gpu,
                    'artifact_dir':str(folder),'artifacts_url':assets.media(folder/'inputs.html')}
                assets.write(folder/'generation.json',result)
                (folder/'raw_output.txt').write_text(raw,encoding='utf-8')
                return result
            except Exception as exc:
                assets.write(folder/'failure.json',{'error':f'{type(exc).__name__}: {exc}','time':time.time()})
                raise

ENGINE = NativeEngine()
