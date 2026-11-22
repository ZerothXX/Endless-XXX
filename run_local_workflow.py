"""Bounded process recovery; never retries because of appearance or a score."""
import argparse, datetime, subprocess, sys, uuid
from pathlib import Path
from local_integrity import read, write, sha
from local_multiref import ROOT

NATIVE_RECOVERABLE = {0xC0000005}


def stage_fingerprints(out):
    return {str(p): sha(p) for p in (out / 'stages').glob('*.json')}


def recoverable(out, controller_code, previous_stages=None):
    if (controller_code & 0xffffffff) in NATIVE_RECOVERABLE: return True
    path = out / 'status.json'
    if not path.exists(): return False
    status = read(path)
    if status.get('stage') != 'failed': return False
    for entry in (out / 'stages').glob('*.json'):
        if previous_stages is not None and previous_stages.get(str(entry)) == sha(entry): continue
        stage = read(entry)
        if stage.get('log') == status.get('log') and (stage.get('exit_code', 0) & 0xffffffff) in NATIVE_RECOVERABLE:
            return True
    return False


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--out', type=Path); p.add_argument('--technical-retries', type=int, choices=[0, 1, 2], default=1)
    a, forwarded = p.parse_known_args()
    if '--help' in forwarded or '-h' in forwarded:
        raise SystemExit(subprocess.run([sys.executable, str(ROOT / 'local_workflow.py'), '--help']).returncode)
    out = (a.out or ROOT / 'output/local_workflows' / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'local_workflow.py'), *forwarded, '--out', str(out)]
    attempts = []
    history = out / 'supervisor_history' / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json')
    for attempt in range(a.technical_retries + 1):
        previous_stages = stage_fingerprints(out)
        code = subprocess.run(command, cwd=ROOT).returncode
        attempts.append({'attempt': attempt + 1, 'controller_exit_code': code, 'finished': datetime.datetime.now().isoformat(),
            'status': read(out / 'status.json') if (out / 'status.json').exists() else None,
            'stages': {p.name: read(p) for p in (out / 'stages').glob('*.json')}})
        retry = code != 0 and attempt < a.technical_retries and recoverable(out, code, previous_stages)
        report = {'attempts': attempts, 'command': command, 'bounded_retry_pending': retry,
            'retry_policy': 'Only native process access violation; same inputs, prompt, caches and seeds. Never based on quality.',
            'quality_status': 'not_evaluated'}
        write(out / 'supervisor.json', report); write(history, report)
        if not retry: raise SystemExit(code)
        print('Resuming the same local task after a native process fault; completed stages retained.', flush=True)


if __name__ == '__main__': main()
