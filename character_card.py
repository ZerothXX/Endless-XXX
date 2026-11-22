"""Versioned character evidence from ckr/ckw. No role/file-name branches."""
import argparse, hashlib, json
from pathlib import Path
from local_multiref import ROOT, read, write, sha

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
PERSON = '''Inspect this character reference. Return JSON with:
{"view":"front|side|back|face|art|unknown", "identity_facts":["visible hair/eye facts"],
"garment_facts":["visible construction/material/panel facts"],
"garment_boxes":[{"label":"blouse/skirt/sleeve/outerwear/etc", "box":[x0,y0,x1,y1]}],
"carried_objects":["camera/weapon/fan/etc if visible"], "uncertainties":["..."]}.
Give up to four tight boxes of visible CLOTHING, excluding face, hair, bare legs, skin and background.
Do not box the whole person. A face-only reference can have no garment boxes.
Do not guess a character name. Coordinates must be normalized to 0..1000.'''
ORNAMENT = '''Inspect this isolated design asset. Return JSON:
{"category":"flat_pattern|physical_ornament|functional_object|unknown", "description":"short exact visible description",
"shape_facts":["outer contour, inner structure, colors, petal count only if certain"],
"suggested_use":"flat surface pattern / small wearable attachment / optional functional prop / unknown",
"uncertainties":["..."]}. Distinguish an actual camera/fan from a flower pattern.
Never recommend copying a large handheld object into every target. Do not guess a character name.'''
SUMMARY = '''These images depict one character, with several viewpoints and possibly a face closeup/artwork.
Return JSON {"identity":"recurring hair, face design, eye colors and gradients",
"garment":"recurring garment construction, layers, front/back differences and materials",
"palette_words":["dominant garment colors only"],
"optional_props":["carried objects, not mandatory costume"],"uncertainties":["..."]}.
Use only visible evidence. Do not use poses/background as identity. Do not infer a real person or character name.
Preserve the difference between skin/hair colors and clothing colors. Keep text concise, total under 240 words.'''


def prepare(role,reuse=False,recipe=None):
    directory = ROOT / 'dataset' / role
    inventory = []
    for kind in ('ckr', 'ckw'):
        files = sorted(p for p in (directory/kind).iterdir() if p.suffix.lower() in EXTENSIONS)
        if not files: raise ValueError(f'Missing {kind}: {directory}')
        inventory += [{'kind': kind, 'path': str(p.resolve()), 'sha256': sha(p)} for p in files]
    dependencies={name:sha(ROOT/name) for name in ('character_card.py','local_color_statistics.py','local_clothing_masks.py','local_vision_tasks.py','local_shape_signature.py','local_palette_prompt.py')}
    recipe=recipe or {'kind':'staged API; actual model and precision recorded in each inherited or fresh evidence file'}
    digest = hashlib.sha256(json.dumps([inventory, dependencies, recipe, 1], sort_keys=True).encode()).hexdigest()[:16]
    version = directory / 'card' / 'versions' / digest
    if (version/'card.json').exists():
        existing=read(version/'card.json')
        for asset in [*existing['source_assets'],*existing['derived_assets']]:
            if sha(asset['path'])!=asset['sha256']:raise ValueError('Immutable card evidence changed; refusing overwrite')
        write(directory/'card/build.json',{'version':str(version.resolve()),'source_digest':digest,'status':'existing immutable version'})
        return version
    write(version/'inventory.json', {'schema_version': 1, 'role': role, 'assets': inventory,
        'source_digest': digest, 'builder_sha256': sha(__file__),'dependencies':dependencies,'recipe':recipe,'status': 'prepared'})
    frozen = version / 'workers' / (sha(__file__)+'.py'); frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(Path(__file__).read_bytes())
    for name,digest_value in dependencies.items():
        (version/'workers'/(digest_value+'.py')).write_bytes((ROOT/name).read_bytes())
    jobs = []
    for i, asset in enumerate(inventory):
        jobs.append({'id': f'asset_{i:02d}', 'images': [asset['path']],
            'prompt': PERSON if asset['kind']=='ckr' else ORNAMENT, 'max_new_tokens': 750})
    jobs.append({'id': 'identity_summary', 'images': [x['path'] for x in inventory if x['kind']=='ckr'],
                 'prompt': SUMMARY, 'max_new_tokens': 850})
    write(version/'vision_jobs.json', {'jobs': jobs})
    if reuse and (directory/'card/current.json').exists():
        import shutil
        previous=Path(read(directory/'card/current.json')['card']).parent
        if read(previous/'inventory.json')['assets']!=inventory:raise ValueError('Inputs differ; evidence reuse refused')
        copied=[]
        for name in ('vision','masks_grounded'):
            if (previous/name).exists():
                shutil.copytree(previous/name,version/name,dirs_exist_ok=True);copied.append(name)
        origin=previous
        while not (origin/'masks_grounded_report.json').exists() and (origin/'inherited_evidence.json').exists():
            origin=Path(read(origin/'inherited_evidence.json')['source_version'])
        if (origin/'masks_grounded_report.json').exists():
            shutil.copy2(origin/'masks_grounded_report.json',version/'inherited_masks_grounded_report.json')
        write(version/'inherited_evidence.json',{'source_version':str(previous.resolve()),
            'segmentation_report_origin':str(origin.resolve()),'copied':copied,
            'reason':'identical source asset SHAs; predictions retain original model/worker provenance, not rerun',
            'vision_status':'uncalibrated predictions; strong text constraints withheld'})
    write(directory/'card'/'build.json', {'version': str(version.resolve()), 'source_digest': digest,
        'status': 'prepared; predictions require validation'})
    print(version, flush=True)
    return version


def safe_boxes(value):
    boxes = []
    for item in (value or {}).get('garment_boxes', []):
        if isinstance(item,list):item={'box':item,'label':'unlabeled clothing'}
        if not isinstance(item,dict):continue
        box = item.get('box')
        if not isinstance(box, list) or len(box)!=4: continue
        try: x0,y0,x1,y1 = [max(0, min(1000, float(v))) for v in box]
        except (TypeError, ValueError): continue
        if x1>x0 and y1>y0 and (x1-x0)*(y1-y0) >= 1000:
            boxes.append({'label': item.get('label', 'clothing'), 'box': [x0,y0,x1,y1]})
    return boxes


def compile_card(role):
    """Requires saved segmentation masks; each eligible image contributes equal mass."""
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    from local_color_statistics import rgb2lab, lab2rgb, cluster_colors
    build = read(ROOT/'dataset'/role/'card'/'build.json'); version=Path(build['version'])
    if (version/'card.json').exists():
        existing=read(version/'card.json')
        for asset in [*existing['source_assets'],*existing['derived_assets']]:
            if sha(asset['path'])!=asset['sha256']:raise ValueError('Immutable card evidence changed; refusing overwrite')
        write(ROOT/'dataset'/role/'card/current.json',{'version':existing['version'],'card':str((version/'card.json').resolve()),'status':existing['status']})
        print('Kept existing immutable card',role,existing['version'],flush=True);return
    inv=read(version/'inventory.json'); predictions=[]; samples=[]; per_image=[]
    rng=np.random.default_rng(20261107)
    for i, asset in enumerate(inv['assets']):
        assert sha(asset['path'])==asset['sha256'], 'Reference changed during card build'
        result=read(version/'vision'/(f'asset_{i:02d}.json'))
        value=result['parsed'] or {}; predictions.append({'asset':asset, 'prediction':value,
            'evidence':str((version/'vision'/(f'asset_{i:02d}.json')).resolve())})
        if asset['kind']!='ckr': continue
        mask_path=version/'masks_grounded'/(f'asset_{i:02d}.png')
        if not mask_path.exists(): continue
        with Image.open(asset['path']) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
        with Image.open(mask_path) as m: mask=np.array(m.convert('L').resize(im.size))>127
        colors=np.array(im)[mask]
        if len(colors)<100: continue
        # Equal per-reference sampling avoids letting artwork resolution dominate.
        pick=colors[rng.choice(len(colors), 6000, replace=len(colors)<6000)]/255.
        samples.append(pick); per_image.append({'path':asset['path'],'source_sha256':asset['sha256'],
            'mask':str(mask_path.resolve()),'mask_sha256':sha(mask_path),
            'foreground_pixels':len(colors), 'view':value.get('view','unknown')})
    if not samples: raise ValueError('No validated clothing masks; refusing an unmasked background palette')
    pixels=np.concatenate(samples);lab=rgb2lab(pixels[:,None,:])[:,0,:]
    centers,labels=cluster_colors(lab)
    rgb=np.clip(lab2rgb(centers[:,None,:])[:,0,:]*255,0,255).round().astype('uint8')
    counts=np.bincount(labels,minlength=8);order=np.argsort(-counts)
    palette=[{'rgb':rgb[i].tolist(),'fraction':float(counts[i]/counts.sum())} for i in order]
    write(version/'palette.json', {'method':'equal-per-image masked clothing pixels; CIELAB KMeans k=8',
        'references':per_image, 'colors':palette, 'mask_status':'local Grounding DINO detections + local SAM2; face-only references excluded; visually inspect overlays',
        'scope':'garment palette estimate; local segmentation may contain errors, inspect overlays'})
    summary=read(version/'vision/identity_summary.json').get('parsed') or {}
    # Complete references remain intact. Ordering is data-driven by locally observed view.
    priority={'front':0,'side':1,'back':2,'face':3,'art':4,'unknown':5}
    people=sorted([x for x in predictions if x['asset']['kind']=='ckr'],
        key=lambda x:priority.get(x['prediction'].get('view','unknown'),5))
    motifs=[x for x in predictions if x['asset']['kind']=='ckw']
    required=[x for x in motifs if x['prediction'].get('category') in ('flat_pattern','physical_ornament')]
    optional=[x for x in motifs if x not in required]
    # Physical samples are evidence images, never overlays pasted onto final outputs.
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(1024,768),'white');draw=ImageDraw.Draw(board);x=0
    for item in palette:
        end=min(1024,x+round(item['fraction']*1024));draw.rectangle((x,0,end,120),fill=tuple(item['rgb']));x=end
    draw.text((12,130),'Automatic clothing palette + exact source design evidence',fill='black',font=font)
    for i,item in enumerate(required):
        with Image.open(item['asset']['path']) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
            pic=ImageOps.contain(bg.convert('RGB'),(min(960//max(1,len(required)),480),530))
        slot=1024/max(1,len(required));cx=int(i*slot+(slot-pic.width)/2)
        board.paste(pic,(cx,180+(530-pic.height)//2))
    board.save(version/'material_board.png')
    # Profiles retain identical complete evidence; their generative utility must be tested.
    sheet=Image.new('RGB',(1536,1024),'white')
    for i,item in enumerate(people):
        with Image.open(item['asset']['path']) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
            slot=1536/len(people);pic=ImageOps.contain(bg.convert('RGB'),(max(1,int(slot)-6),1000))
        sheet.paste(pic,(int(i*slot+(slot-pic.width)/2),(1024-pic.height)//2))
    sheet.save(version/'identity_board.png')
    dominant=Image.new('RGB',(1024,768),'white');draw=ImageDraw.Draw(dominant);x=0
    for item in palette:
        end=min(1024,x+round(item['fraction']*1024));draw.rectangle((x,0,end,510),fill=tuple(item['rgb']));x=end
    for i,item in enumerate(required):
        with Image.open(item['asset']['path']) as raw:
            pic=ImageOps.contain(ImageOps.exif_transpose(raw).convert('RGBA'),(180,180))
            bg=Image.new('RGBA',pic.size,'white');bg.alpha_composite(pic)
        dominant.paste(bg.convert('RGB'),(int((i+.5)*1024/len(required)-pic.width/2),550))
    dominant.save(version/'material_palette_dominant.png')
    from local_shape_signature import shape_signature
    write(version/'shape_evidence.json',{'method':'alpha contour angular spectrum; not universal petal counting',
        'assets':[{'path':x['asset']['path'],'sha256':x['asset']['sha256'],
                   'signature':shape_signature(x['asset']['path'])} for x in motifs]})
    derivatives=[version/name for name in ('palette.json','shape_evidence.json','identity_board.png','material_board.png','material_palette_dominant.png')]
    derivatives+=sorted((version/'vision').glob('*.json'))
    derivatives+=[version/name for name in ('inventory.json','vision_jobs.json','masks_grounded_report.json','inherited_masks_grounded_report.json','inherited_evidence.json') if (version/name).exists()]
    derivatives += [Path(x['mask']) for x in per_image]
    card={'schema_version':1,'role':role,'version':inv['source_digest'],'status':'experimental; not quality accepted',
        'source_assets':inv['assets'],'identity_references':[x['asset']['path'] for x in people],
        'material_references':[str((version/'material_board.png').resolve())],
        'profiles':{'human_complete':[x['asset']['path'] for x in people],
            'human_sheet':[str((version/'identity_board.png').resolve())],
            'material_motif_dominant':[str((version/'material_board.png').resolve())],
            'material_palette_dominant':[str((version/'material_palette_dominant.png').resolve())],
            'material_isolated':[x['asset']['path'] for x in required]},
        'profile_policies':{'material_isolated':'numeric garment palette with deterministic color wording; original isolated motifs only'},
        'derived_assets':[{'path':str(p.resolve()),'sha256':sha(p)} for p in derivatives],
        # Uncalibrated VLM facts are retained for inspection, not promoted into instructions.
        'identity_constraints':'','design_constraints':'',
        'candidate_identity_description':summary,
        'text_policy':'Uncalibrated descriptions are evidence only. Full reference pixels provide identity/design.',
        'ornaments':required,'optional_props':optional,'palette':palette,
        'uncertainties':summary.get('uncertainties',[]),'vision_predictions':predictions,
        'provenance':'local image statistics and offline VLM/SAM2 predictions; no assistant-authored role traits',
        'policy':{'human':'identity and costume evidence only; preserve source pose/expression',
            'object':'translate palette/patterns/material; keep functional shape',
            'animal':'natural animal preserved; narrow attached accessories only',
            'optional_props':'not automatically mandatory; include only when task explicitly requests a functional prop'},
        'online_inference':False}
    write(version/'card.json',card)
    write(ROOT/'dataset'/role/'card'/'current.json', {'version':inv['source_digest'],
        'card':str((version/'card.json').resolve()),'status':card['status']})
    print('Compiled',role,'palette references',len(per_image),'motifs',len(required),'optional',len(optional),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','inspect'])
    p.add_argument('--roles',nargs='+',required=True);p.add_argument('--reuse-from-current',action='store_true')
    p.add_argument('--recipe-json',type=Path);a=p.parse_args()
    for role in a.roles:
        if Path(role).name!=role or role in ('.','..'):raise ValueError('Invalid role folder')
        if a.stage=='inspect':
            pointer=read(ROOT/'dataset'/role/'card/current.json');card=read(pointer['card'])
            print(json.dumps({'role':role,'version':pointer['version'],'status':card['status'],
                'source_count':len(card['source_assets']),'palette':card['palette'],
                'profiles':card.get('profiles',{}),'optional_prop_count':len(card['optional_props']),
                'directory':str(Path(pointer['card']).parent)},ensure_ascii=False,indent=2))
        elif a.stage=='prepare':prepare(role,a.reuse_from_current,read(a.recipe_json) if a.recipe_json else None)
        else:compile_card(role)


if __name__=='__main__':main()
