"""Local production orchestration. Completion means delivered, never quality accepted.

Exact existing inputs retain the frozen recipe. Fresh tasks run local PE, BF16
encoding and the existing NF4 renderer. No critic, quality score or review gate.
"""
import argparse, contextlib, datetime, hashlib, html, json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
from local_integrity import read, write, sha, snapshot, validate_files
from local_multiref import ROOT, DEFAULT_MODEL
from local_edit_runtime import build_plan, load_role_card

PYTHON = Path(os.environ.get('ENDLESS_PYTHON', sys.executable))
BASELINE = ROOT / 'histories/quality_20261116_stability'
REWRITER = ROOT / 'local_prompt_planner.py'


@contextlib.contextmanager
def run_lock(out):
    import msvcrt
    with (out / '.run.lock').open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0: f.write(b'0'); f.flush()
        f.seek(0)
        try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise RuntimeError('This run directory is already in use')
        try: yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def run_stage(out, name, script, args, expected=None):
    command = [str(PYTHON), '-u', str(script), *map(str, args)]
    status_path = out / 'stages' / (name + '.json')
    if expected and status_path.exists() and expected.exists():
        previous = read(status_path)
        if previous.get('exit_code') == 0 and previous.get('command') == command and previous.get('artifact_sha256') == sha(expected):
            print('Reusing completed stage:', name, flush=True); return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logfile = out / 'logs' / (name + '_' + stamp + '.log'); logfile.parent.mkdir(exist_ok=True)
    write(out / 'status.json', {'running': True, 'stage': name, 'log': str(logfile), 'quality_status': 'not_evaluated'})
    print('Local stage:', name, flush=True)
    with logfile.open('wb') as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}).returncode
    result = {'command': command, 'exit_code': code, 'log': str(logfile), 'finished': datetime.datetime.now().isoformat()}
    if code == 0 and expected:
        if not expected.is_file(): raise RuntimeError('Stage did not produce its artifact: ' + name)
        result['artifact_sha256'] = sha(expected)
    write(status_path, result)
    if code: raise RuntimeError('Local stage failed; see ' + str(logfile))


def ensure_card(character, out):
    folder = ROOT / 'dataset' / character
    from character_card import EXTENSIONS
    current_assets = {str(p.resolve()): sha(p) for kind in ['ckr', 'ckw']
                      for p in (folder / kind).glob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if not any(str(Path(x).parent) == str((folder / 'ckr').resolve()) for x in current_assets):
        raise ValueError('No character references in dataset/' + character + '/ckr')
    valid = False
    try:
        path, card = load_role_card(folder)
        valid = bool(card.get('source_assets')) and {str(Path(a['path']).resolve()): a['sha256'] for a in card['source_assets']} == current_assets
    except (ValueError, FileNotFoundError, KeyError): pass
    if not valid:
        card_workers = [ROOT / 'build_role_cards.py', ROOT / 'local_vision_tasks.py', ROOT / 'local_clothing_masks.py']
        missing = [p.name for p in card_workers if not p.is_file()]
        if missing:
            raise RuntimeError('No current role card matches dataset/' + character + '/ckr. This release does not include the role-card vision/mask workers; missing: ' + ', '.join(missing) + '. Supply a versioned local role card before generation.')
        run_stage(out, 'build_card', ROOT / 'build_role_cards.py', ['--roles', character])
        path, card = load_role_card(folder)
        if {str(Path(a['path']).resolve()): a['sha256'] for a in card['source_assets']} != current_assets:
            raise ValueError('Role card does not match current reference assets')
    return path, card


def preserved_recipe(case, model):
    if case['kind'] != 'human' or not (BASELINE / 'baseline.json').is_file(): return None
    b = read(BASELINE / 'baseline.json')
    if Path(b['configuration']['model']).resolve() != model.resolve(): return None
    plans = read(BASELINE / 'run/plan.json')['cases']
    for old in plans:
        if old['role'] != case['role'] or Path(old['source']).resolve() != Path(case['source']).resolve(): continue
        if old['references'] != case['references'] or (old['width'], old['height']) != (case['width'], case['height']): continue
        # Refuse drift instead of silently reusing a visually preferred stale recipe.
        from local_integrity import verify
        verify(BASELINE)
        cache = next(e for e in b['encodings'] if Path(e['path']).name == old['id'] + '.pt')
        return old, cache
    return None


def prepare(character, inputs, kind, seeds, out, fresh, model):
    if Path(character).name != character or character in ['.', '..'] or '/' in character or '\\' in character:
        raise ValueError('Invalid character folder')
    if len({str(p.resolve()) for p in inputs}) != len(inputs): raise ValueError('Duplicate input paths')
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds): raise ValueError('Invalid or duplicate seeds')
    if not (model / 'model_index.json').is_file(): raise FileNotFoundError('Local generator model is missing: ' + str(model))
    card_path, card = ensure_card(character, out)
    profile = 'human_complete' if kind == 'human' else 'material_isolated'
    raw = build_plan(character, inputs, kind, out / 'input_plan', 768, design_card=card_path, reference_profile=profile)
    cases = read(raw)['cases']
    request = {'character': character, 'kind': kind, 'seeds': seeds, 'fresh': fresh, 'model': str(model.resolve()),
        'inputs': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in inputs],
        'card': str(card_path), 'card_sha256': sha(card_path),
        'references': [{'path': p, 'sha256': sha(p)} for p in cases[0]['references']],
        'workers': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'local_integrity.py', ROOT / 'local_multiref.py', ROOT / 'local_edit_runtime.py', ROOT / 'local_palette_prompt.py', ROOT / 'local_quant_offload.py', ROOT / 'gpu_slot.py', ROOT / 'safetensors_pread.py', REWRITER]},
        'configuration': {'encoder_precision': 'bf16', 'generator_precision': 'nf4', 'steps': 40, 'reference_resolution': 512,
                          'cpu_offload': True, 'kv_cache_cpu': True, 'trim_cache': True},
        'automatic_quality_check': False, 'quality_status': 'not_evaluated'}
    request_path = out / 'request.json'
    if request_path.exists():
        if read(request_path) != request: raise ValueError('Inputs, card, parameters or workers changed; use a new run directory')
        manifest = read(out / 'manifest.json')
        if sha(request_path) != manifest['request_sha256']: raise ValueError('Saved request content changed')
        validate_files(manifest['snapshots'])
        return manifest
    retained = []; pending = []; routing = []; snapshots = {}
    def preserve(path):
        snapshots[str(Path(path).resolve())] = snapshot(path, out)
    for path in [*(p for p in inputs), card_path, *[Path(p) for p in cases[0]['references']],
                 Path(__file__), ROOT / 'local_integrity.py', ROOT / 'local_multiref.py', ROOT / 'local_edit_runtime.py', ROOT / 'local_palette_prompt.py', ROOT / 'local_quant_offload.py', ROOT / 'gpu_slot.py', ROOT / 'safetensors_pread.py', REWRITER]: preserve(path)
    for case in cases:
        match = None if fresh else preserved_recipe(case, model)
        if match:
            old, cache = match
            destination = out / 'render/embeddings' / Path(cache['path']).name
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(cache['path'], destination)
            assert sha(destination) == cache['sha256']
            preserve(destination); retained.append(old)
            routing.append({'input': case['source'], 'case_id': old['id'], 'route': 'exact_preserved_recipe',
                            'baseline_sha256': sha(BASELINE / 'baseline.json'), 'cache_sha256': cache['sha256']})
        else:
            digest = hashlib.sha256(json.dumps([case['source'], sha(case['source']), character, kind]).encode()).hexdigest()[:12]
            case['id'] = 'auto_' + digest; pending.append(case)
            routing.append({'input': case['source'], 'case_id': case['id'], 'route': 'fresh_local_planning',
                            'scope': 'No per-image hand-authored facts; quality equivalence to preserved recipes is unproven'})
    write(out / 'pending.json', {'cases': pending}); write(out / 'retained.json', {'cases': retained})
    for p in [out / 'pending.json', out / 'retained.json']: preserve(p)
    write(request_path, request)
    manifest = {'routing': routing, 'snapshots': list(snapshots.values()), 'request_sha256': sha(request_path),
                'quality_status': 'not_evaluated', 'online_inference': False}
    write(out / 'manifest.json', manifest)
    return manifest


def collect_results(out, seeds):
    cases = read(out / 'plan.json')['cases']; rows = []
    expected = {c['id'] + '_s' + str(seed) + '_nf4.json' for c in cases for seed in seeds}
    files = {p.name for p in (out / 'render/records').glob('*.json')}
    if files != expected: raise ValueError('Incomplete or unexpected output records; no silent sample reduction')
    for c in cases:
        for seed in seeds:
            path = out / 'render/records' / (c['id'] + '_s' + str(seed) + '_nf4.json'); r = read(path)
            if r['binding']['case'] != c or r['online_inference'] is not False or r['worker_sha256'] != sha(ROOT / 'local_multiref.py'):
                raise ValueError('Output provenance mismatch')
            if sha(r['output']) != r['sha256']: raise ValueError('Output image changed')
            for asset in r['binding']['inputs']:
                if sha(asset['path']) != asset['sha256']: raise ValueError('Input changed during generation')
            image = snapshot(r['output'], out); snapshot(path, out)
            rows.append({'id': c['id'] + '_s' + str(seed), 'case': c, 'seed': seed, 'output': r['output'],
                         'output_sha256': r['sha256'], 'frozen_output': image['frozen'], 'record': str(path),
                         'quality_status': 'not_evaluated'})
    write(out / 'results.json', {'rows': rows, 'online_inference': False, 'quality_status': 'not_evaluated',
                               'technical_completion': True, 'automatic_quality_check': False})
    return rows


def gallery(out, rows):
    def uri(path): return html.escape(Path(path).resolve().as_uri())
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>本地自动生成结果</title><style>body{font:16px system-ui;margin:24px;background:#f4f6fa;color:#183048}img{max-width:100%;max-height:520px}section{background:white;padding:18px;margin:20px 0}.pair{display:flex;gap:18px}.pair figure{margin:0;flex:1}.refs img{max-width:18%;max-height:300px}p{line-height:1.6}</style><h1>本地自动生成结果</h1><p>自动运行已结束，画质尚未评定。生成后直接交付，不自动评分、不选优，也不等待视觉验收。原图、全部完整参考和各个种子的输出均保留。</p>']
    manifest = read(out / 'manifest.json'); frozen = {e['original']: e['frozen'] for e in manifest['snapshots']}
    routes = {r['input']: r['route'] for r in manifest['routing']}
    for row in rows:
        c = row['case']; route = routes[c['source']]
        title = ('复用已保留配方' if route == 'exact_preserved_recipe' else '本地重新规划；效果待看图确认')
        parts.append('<section><h2>' + html.escape(c['role'] + ' / ' + c['target'] + ' / 种子' + str(row['seed'])) + '</h2><p>' + title + '</p><div class="pair">')
        for label, image in [('原图', frozen.get(c['source'], c['source'])), ('生成结果', row['frozen_output'])]:
            parts.append('<figure><figcaption>' + label + '</figcaption><a href="' + uri(image) + '"><img src="' + uri(image) + '"></a></figure>')
        parts.append('</div><details><summary>全部完整角色参考</summary><div class="refs">')
        parts.extend('<img src="' + uri(frozen.get(p, p)) + '">' for p in c['references'])
        parts.append('</div></details></section>')
    parts.append('</html>'); (out / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


def execute(out, seeds, model):
    retained = read(out / 'retained.json')['cases']; pending = read(out / 'pending.json')['cases']
    if pending:
        compiled = out / 'rewrite/plan.json'
        run_stage(out, 'rewrite', REWRITER, ['--plan', out / 'pending.json', '--out', out / 'rewrite'], compiled)
        rewritten = read(compiled)['cases']
        if {c['id'] for c in rewritten} != {c['id'] + '_pe' for c in pending} or len(rewritten) != len(pending):
            raise ValueError('Local planning failed for some inputs; no silent dropping or generic fallback')
        originals = {c['id'] + '_pe': c for c in pending}
        for c in rewritten:
            if any(c[k] != originals[c['id']][k] for k in ['role', 'target', 'kind', 'source', 'references', 'width', 'height']):
                raise ValueError('Planner changed the input, references or source canvas')
    else: rewritten = []
    plan = out / 'plan.json'; combined = {'cases': retained + rewritten, 'online_inference': False}
    if plan.exists() and read(plan) != combined: raise ValueError('Compiled plan changed during resume')
    write(plan, combined)
    shared = ['--plan', plan, '--out', out / 'render', '--model', model, '--encoder-precision', 'bf16',
        '--encoder-vision-bf16', '--precision', 'nf4', '--steps', 40, '--reference-resolution', 512,
        '--cpu-offload', '--kv-cache-cpu', '--trim-cache', '--seeds', *seeds]
    # Existing records are checked before a resumed renderer can skip them.
    for path in (out / 'render/records').glob('*.json'):
        r = read(path)
        if r['binding']['case'] not in combined['cases'] or sha(r['output']) != r['sha256']:
            raise ValueError('Existing result changed; refusing silent resume')
    run_stage(out, 'encode', ROOT / 'local_multiref.py', ['--stage', 'encode', *shared,
        '--encoder-disk-offload', out / 'encoder_disk'])
    run_stage(out, 'render', ROOT / 'local_multiref.py', ['--stage', 'render', *shared])
    rows = collect_results(out, seeds); gallery(out, rows)
    write(out / 'status.json', {'running': False, 'stage': 'delivered', 'outputs': len(rows),
        'finished': datetime.datetime.now().isoformat(), 'technical_completion': True,
        'quality_status': 'not_evaluated', 'automatic_quality_check': False})
    print('Delivered local outputs:', len(rows), str(out / 'index.html'), flush=True)


def main():
    p = argparse.ArgumentParser(description='本地自动规划与生成，最终效果由人看图检查')
    p.add_argument('--character', required=True); p.add_argument('--input', nargs='+', type=Path, required=True)
    p.add_argument('--kind', choices=['human', 'object', 'animal'], default='human')
    p.add_argument('--seeds', nargs='+', type=int, default=[42]); p.add_argument('--out', type=Path)
    p.add_argument('--fresh', action='store_true', help='独立实验：强制重新规划，不复用旧图条件；不保证效果等同')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); out = (a.out or ROOT / 'output/local_workflows' / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        try:
            prepare(a.character, a.input, a.kind, a.seeds, out, a.fresh, a.model)
            if a.prepare_only:
                write(out / 'status.json', {'running': False, 'stage': 'prepared', 'quality_status': 'not_evaluated'})
                print('Prepared:', out, flush=True)
            else: execute(out, a.seeds, a.model)
        except Exception as exc:
            previous = read(out / 'status.json') if (out / 'status.json').exists() else {}
            write(out / 'status.json', {**previous, 'running': False, 'stage': 'failed', 'error': str(exc), 'quality_status': 'not_evaluated'})
            raise


if __name__ == '__main__': main()
