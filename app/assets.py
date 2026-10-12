"""Read-only predicted scene assets; metadata/geometry projection reused from t50."""
from __future__ import annotations
import functools, hashlib, json, os, sys, threading
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import config

PROJECT = Path(os.environ.get('VTC_ASSETS_ROOT',str(Path(__file__).resolve().parents[2]/'VTC-3D-MLLM')))
sys.path.insert(0, str(PROJECT/'src/t50'))
sys.path.insert(0, str(PROJECT/'runtime/t48_render_deps'))
from model_io import file_sha256
from scene_evidence import load_observations, load_postprocessed_masks, visibility
from metadata_cache import load_scene_metadata
from preparation.render_and_finalize import render_document

LOCK = threading.RLock()
CACHE = Path(config.WEBUI_ROOT)/'results/input_cache'
CACHE.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def media(path):
    rel = Path(path).resolve().relative_to(Path(config.WEBUI_ROOT).resolve())
    return '/api/media/asset?path='+quote(str(rel))

@functools.lru_cache(maxsize=16)
def scene_assets(scene):
    scene = scene.split('/')[-1]
    index = read(Path(os.environ.get('VTC_INVENTORY_INDEX',str(PROJECT/'scene_inventory/index.json'))))
    inventory_path = Path(index[scene]); inventory = read(inventory_path)
    manifest_path = Path(os.environ.get('VTC_CAPTIONS_ROOT',str(PROJECT/'captions')))/scene/'captions_manifest.json'
    manifest = read(manifest_path)
    if manifest['scene_id'] != scene or manifest['inventory_sha256'] != file_sha256(inventory_path):
        raise ValueError('Caption inventory identity mismatch')
    captions = {}
    for artifact in manifest['caption_artifacts']:
        if artifact.get('status') not in {'ready','unobserved'}:
            raise ValueError('Caption not ready: '+str(artifact['display_id']))
        if file_sha256(artifact['path']) != artifact['sha256']:
            raise ValueError('Caption checksum mismatch')
        captions[str(artifact['display_id'])] = read(artifact['path'])['text']
    objects = inventory['objects']
    if set(captions) != {str(x['display_id']) for x in objects}:
        raise ValueError('Captions and canonical object IDs differ')
    text = '\n\n'.join(f"Object {o['display_id']} ({o.get('class_name') or 'object'}): {captions[str(o['display_id'])]}" for o in objects)
    # Avoid ambiguous camera-relative left/right. Geometry remains in axis-aligned meters.
    relations = ['Geometric references: ScanNet axis-aligned world coordinates, meters; Z points upward. Above/below compare object centers. Distances are center-to-center Euclidean distances. Image left/right may change between frames.']
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            ca, cb = np.array(a['center_m']), np.array(b['center_m'])
            distance = float(np.linalg.norm(ca-cb))
            if distance > 1.5: continue
            dz = float(ca[2]-cb[2]); phrase = 'at approximately the same height as'
            if dz > .15: phrase = 'above'
            if dz < -.15: phrase = 'below'
            relations.append(f"Object {a['display_id']} is {phrase} object {b['display_id']}; center distance {distance:.2f} m.")
    return {'scene': scene, 'inventory_path': str(inventory_path), 'inventory_sha256': file_sha256(inventory_path),
            'caption_manifest': str(manifest_path), 'caption_manifest_sha256': file_sha256(manifest_path),
            'description_text': text, 'relation_text': '\n'.join(relations), 'objects': objects,
            'relation_rule': 'all unordered canonical pairs within 1.5m, including same-class; no observer left/right; not a nearest/farthest candidate filter',
            'gt_fields_consumed': [], 'inventory': inventory}

@functools.lru_cache(maxsize=2)
def scene_geometry(scene):
    inv = scene_assets(scene)['inventory']
    source = inv['source_artifacts']
    if file_sha256(source['cloud']) != source['cloud_sha256'] or file_sha256(source['predicted_masks']) != source['mask_sha256']:
        raise ValueError('Predicted mask/cloud checksum mismatch')
    xyz, _ = load_observations(source['cloud'])
    masks = load_postprocessed_masks(source['predicted_masks'], len(xyz))
    by_id = {str(m['source_index']): m['indices'] for m in masks}
    selected = {str(o['display_id']): by_id[str(o['display_id'])] for o in inv['objects']}
    meta = load_scene_metadata(Path(config.REPO_ROOT)/'data/embodiedscan', scene)
    return xyz, selected, meta

def numbered_frame(scene, source_path):
    scene = scene.split('/')[-1]; source_path = Path(source_path)
    identity = {'drawing_version': 1, 'source_sha256': file_sha256(source_path),
                'inventory_sha256': scene_assets(scene)['inventory_sha256'], 'label_limit': 32,
                'depth_sha256': file_sha256(source_path.with_suffix('.png')),
                'pose_sha256': file_sha256(source_path.with_suffix('.txt'))}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = CACHE/'numbered'/digest; manifest = folder/'manifest.json'
    with LOCK:
        if manifest.exists():
            result = read(manifest)
            if file_sha256(result['path']) != result['sha256']: raise ValueError('Marked image cache damaged')
            return result
        xyz, masks, meta = scene_geometry(scene)
        with Image.open(source_path) as im: image = im.convert('RGB')
        depth = np.asarray(Image.open(source_path.with_suffix('.png')))
        world_from_cam = np.asarray(meta['axis_align_matrix']) @ np.loadtxt(source_path.with_suffix('.txt'))
        uv, visible = visibility(xyz, world_from_cam, np.asarray(meta['cam2img']), np.asarray(meta['depth_cam2img']), image.size, depth)
        candidates = []
        for display_id, indices in masks.items():
            points = uv[indices[visible[indices]]]
            if len(points) < 20: continue
            median = np.median(points, axis=0)
            anchor = points[np.argmin(((points-median)**2).sum(axis=1))]
            candidates.append((len(points), display_id, anchor))
        candidates.sort(key=lambda x: (-x[0], int(x[1])))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(24, round(image.width*.026)))
        draw = ImageDraw.Draw(image); boxes = []; labels = []; omitted = []
        for count, display_id, anchor in candidates:
            box = draw.textbbox((0,0), display_id, font=font)
            w, h = box[2]-box[0]+10, box[3]-box[1]+8
            # Keep labels at actual visible object anchors. Omit collisions, never drift into another object.
            x = min(max(0, round(float(anchor[0])-w/2)), image.width-w)
            y = min(max(0, round(float(anchor[1])-h/2)), image.height-h)
            rect = (x, y, x+w, y+h)
            overlap = any(not (rect[2]+3 < b[0] or rect[0]-3 > b[2] or rect[3]+3 < b[1] or rect[1]-3 > b[3]) for b in boxes)
            if overlap or len(labels) >= 32:
                omitted.append(display_id); continue
            draw.rounded_rectangle(rect, radius=3, fill='white', outline='#bd1717', width=2)
            draw.text((x+5-box[0], y+4-box[1]), display_id, font=font, fill='#bd1717')
            boxes.append(rect); labels.append({'display_id': display_id, 'anchor_px': anchor.tolist(), 'visible_points': count, 'label_box': list(rect)})
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'numbered.png'; image.save(path, format='PNG')
        result = {'path': str(path), 'sha256': file_sha256(path), 'url': media(path), 'source_path': str(source_path),
                  'identity': identity, 'labels': labels, 'omitted_collision_or_budget': omitted,
                  'rgb_size': list(image.size), 'calibration_assumption': 'upstream common RGB/depth camera coordinates',
                  'drawing': 'visible mask-point anchor, no leader lines; exact source resolution; not every object is visible in every frame', 'gt_fields_consumed': []}
        write(manifest, result)
        return result

def glyph_pages(document):
    with LOCK:
        result = render_document(document, PROJECT/'third_party/Glyph', CACHE/'glyph')
    return {**result, 'pages': [{**p, 'url': media(p['path'])} for p in result['pages']]}

def processor_input(row, max_pixels):
    """Save exact RGB supplied to native processor; never center-crop a frame."""
    identity = {'source_sha256':row['sha256'], 'max_pixels':max_pixels, 'version':1}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path = CACHE/'processor_rgb'/f'{digest}.png'
    with LOCK:
        with Image.open(row['path']) as original:
            image = original.convert('RGB'); source_size=list(image.size)
            if image.width*image.height > max_pixels:
                scale=(max_pixels/(image.width*image.height))**.5
                image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.BICUBIC)
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True); image.save(path,format='PNG')
            result={**row,'source_path':row['path'],'source_sha256':row['sha256'],'source_url':row['url'],
                    'path':str(path),'sha256':file_sha256(path),'url':media(path),'source_size':source_size,
                    'processor_input_size':list(image.size),'processor_input_RGB_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        return result
