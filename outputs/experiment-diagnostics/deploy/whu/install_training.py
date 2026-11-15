"""Install the prepared training wheel set on an allocated CPU node, with no network."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import venv


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run through the CPU Slurm setup job')
    root = Path.cwd().resolve()
    manifest = json.loads((root / 'training-manifest.json').read_text())
    files = {item['file']: item for item in manifest['files']}
    if any(Path(name).name != name or not name.endswith('.whl') for name in files):
        raise ValueError('Unexpected wheel names')
    wheelhouse = root / 'state/training-wheelhouse'
    wheelhouse.mkdir(exist_ok=True)
    with tarfile.open(root / 'training-wheels.tar') as archive:
        members = archive.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('Unexpected archive members')
        for member in members:
            target = wheelhouse / member.name
            if not member.isfile() or target.is_symlink():
                raise ValueError('Only regular wheel files are permitted')
        archive.extractall(wheelhouse, members=members, filter='data')
    for name, item in files.items():
        path = wheelhouse / name
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Wheel hash mismatch: ' + name)
    model_manifest = json.loads((root / 'model-manifest.json').read_text())
    for item in model_manifest['files']:
        if Path(item['file']).name != item['file']:
            raise ValueError('Unexpected model file name')
        path = root / 'models/Qwen2.5-0.5B-Instruct' / item['file']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Model hash mismatch: ' + item['file'])
    target = root / 'state/training-cu121-py312'
    if target.is_symlink() or not target.resolve().is_relative_to(root / 'state'):
        raise ValueError('Training environment must stay in the project state directory')
    marker = target / '.deerflow-training.json'
    if target.exists() and any(target.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('owner') != 'deerflow-whu-training-v1':
            raise ValueError('Refusing to modify an environment owned by another installation')
    venv.EnvBuilder(with_pip=True).create(target)
    marker.write_text(json.dumps({'owner':'deerflow-whu-training-v1'}))
    python = target / 'bin/python'
    environment = {k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME','DIAG_API_KEY'}}
    environment.update(PIP_NO_INDEX='1', HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(wheelhouse),
                    '--require-hashes', '--only-binary=:all:', '-r', str(root/'training.lock'),
                    '--report', str(root/'state/training-offline-install.json')], check=True, env=environment)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True, env=environment)
    with (root / 'state/training-freeze.txt').open('w') as stream:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True, stdout=stream, env=environment)
    receipt = {'slurm_job_id':os.environ['SLURM_JOB_ID'], 'status':'training_environment_installed',
               'python':str(python), 'gpu_allocated':False, 'model_files_hash_verified':True,
               'wheel_count':len(files), 'network_used_by_installer':False,
               'model_revision':model_manifest['revision'], 'model_weights_sha256':model_manifest['weights_sha256']}
    (root / 'state/training-installation.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
