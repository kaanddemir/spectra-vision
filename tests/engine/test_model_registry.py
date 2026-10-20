import hashlib
import json
from pathlib import Path

import pytest


def registry(tmp_path):
    rows = []
    for role, shape in [('detector', [1, 3, 640, 640]), ('segmenter', [1, 3, 384, 512])]:
        p = tmp_path / (role + '.onnx')
        p.write_bytes(role.encode())
        rows.append(dict(role=role, file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         input=shape, licence='test', export_script='test.py', source_checkpoint='test'))
    (tmp_path / 'models.json').write_text(json.dumps({'models': rows}))
    return tmp_path


def test_registry_resolves_verified_specs(tmp_path):
    from spectra.engine.models import default_specs
    detector, segmenter = default_specs(registry(tmp_path))
    assert detector.model_path == str((tmp_path / 'detector.onnx').resolve())
    assert detector.input_size == 640
    assert segmenter.model_path == str((tmp_path / 'segmenter.onnx').resolve())
    assert segmenter.interval_s == 0.2


@pytest.mark.parametrize('problem', ['missing', 'corrupt', 'traversal', 'duplicate', 'shape'])
def test_registry_refuses_invalid_entries(tmp_path, problem):
    from spectra.engine.models import default_specs
    root = registry(tmp_path)
    data = json.loads((root / 'models.json').read_text())
    if problem == 'missing':
        (root / 'detector.onnx').unlink()
    elif problem == 'corrupt':
        (root / 'detector.onnx').write_bytes(b'changed')
    elif problem == 'traversal':
        data['models'][0]['file'] = '../detector.onnx'
    elif problem == 'duplicate':
        data['models'].append(data['models'][0])
    else:
        data['models'][0]['input'] = [1, 3, 100, 200]
    (root / 'models.json').write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        default_specs(root)


def test_only_registry_is_unignored():
    import subprocess
    root = Path(__file__).resolve().parents[2]
    def ignored(name):
        return subprocess.run(['git', 'check-ignore', '--quiet', name], cwd=root).returncode == 0
    assert not ignored('models/engine/models.json')
    assert ignored('models/engine/dfine_n_coco_640.onnx')
    assert ignored('models/engine/dfine_s_coco_640.onnx')
    assert ignored('models/engine/rfdetr_s_coco_512.onnx')


def test_shipped_registry_entries_are_valid():
    from spectra.engine.models import ModelEntry
    root = Path(__file__).resolve().parents[2]
    rows = json.loads((root / 'models' / 'engine' / 'models.json').read_text())['models']
    entries = [ModelEntry.model_validate(row) for row in rows]
    assert sorted(e.role for e in entries) == ['detector', 'segmenter']
    assert all(e.source_url and e.source_sha256 for e in entries)
