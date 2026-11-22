"""Portable input, artifact, and frozen-baseline integrity helpers."""
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def snapshot(path, output_dir):
    path = Path(path).resolve()
    digest = sha(path)
    destination = Path(output_dir) / 'frozen/files' / (digest + path.suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    if sha(destination) != digest:
        raise ValueError('Corrupt frozen copy: ' + str(destination))
    return {'original': str(path), 'frozen': str(destination), 'sha256': digest,
            'bytes': path.stat().st_size}


def validate_files(entries):
    for entry in entries:
        for key in ('original', 'frozen'):
            if sha(entry[key]) != entry['sha256']:
                raise ValueError('Frozen baseline changed: ' + entry[key])


def verify(output_dir):
    output_dir = Path(output_dir)
    baseline = read(output_dir / 'baseline.json')
    validate_files(baseline['files'])
    assert sha(output_dir / 'run/plan.json') == baseline['plan_sha256']
    assert sha(output_dir / 'protocol.json') == baseline['protocol_sha256']
    for entry in baseline['encodings']:
        assert sha(entry['path']) == entry['sha256'], 'Prompt cache changed'
    for entry in baseline['model_inventory']:
        path = Path(entry['path'])
        stat = path.stat()
        assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns'], 'Model stat drift'
    write(output_dir / 'baseline_verification.json', {
        'verified': True,
        'files': len(baseline['files']),
        'embedding_copies': len(baseline['encodings']),
        'model_stat_checked': True,
        'model_full_content_checked': False,
        'quality_accepted': False,
    })
