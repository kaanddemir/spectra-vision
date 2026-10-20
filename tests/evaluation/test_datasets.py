import hashlib
import json
from pathlib import Path

import pytest

from evaluation.datasets import Manifest, clips, load_manifest


def metadata(**kw):
    return dict(id='new-source', filename='sample.bin', bytes=3, role='traffic', licence='test',
                sha256=hashlib.sha256(b'abc').hexdigest(), **kw)


def test_verified_access_is_manifest_driven_and_read_only(tmp_path):
    source = tmp_path / 'sample.bin'
    source.write_bytes(b'abc')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(manifest_version='1.0', clips=[metadata(assumed_fps=17, split='validation')],
                                       total_bytes=3, provenance='preserved')))
    m = load_manifest(manifest)
    [clip] = list(clips(m, tmp_path))
    assert clip.path == source and clip.metadata.assumed_fps == 17
    assert clip.metadata.split == 'validation' and m.model_extra['provenance'] == 'preserved'
    assert source.read_bytes() == b'abc'


@pytest.mark.parametrize('change', [dict(sha256='bad'), dict(sha256=None), dict(bytes=0), dict(assumed_fps=0),
                                    dict(filename='../outside'), dict(filename='/outside'), dict(split='invalid')])
def test_invalid_metadata_is_rejected(change):
    row = metadata(); row.update(change)
    with pytest.raises(ValueError):
        Manifest(clips=[row])


@pytest.mark.parametrize('problem', ['size', 'hash', 'missing', 'symlink'])
def test_integrity_failures_refuse_access(tmp_path, problem):
    root = tmp_path / 'source'; root.mkdir()
    p = root / 'sample.bin'
    if problem == 'size': p.write_bytes(b'ab')
    if problem == 'hash': p.write_bytes(b'xyz')
    if problem == 'symlink':
        outside = tmp_path / 'outside'; outside.write_bytes(b'abc'); p.symlink_to(outside)
    with pytest.raises((ValueError, FileNotFoundError)):
        list(clips(Manifest(clips=[metadata()]), root))


def test_sha1_fallback_and_sha256_preference(tmp_path):
    (tmp_path / 'sample.bin').write_bytes(b'abc')
    row = metadata(); row.pop('sha256'); row['sha1'] = hashlib.sha1(b'abc').hexdigest()
    assert len(list(clips(Manifest(clips=[row]), tmp_path))) == 1
    row.update(sha256='0' * 64)
    with pytest.raises(ValueError, match='hash mismatch'):
        list(clips(Manifest(clips=[row]), tmp_path))


def test_duplicates_totals_and_proposed_manifests():
    with pytest.raises(ValueError, match='unique'):
        Manifest(clips=[metadata(), metadata()])
    with pytest.raises(ValueError, match='total_bytes'):
        Manifest(clips=[metadata()], total_bytes=4)
    root = Path(__file__).resolve().parents[2]
    assert len(load_manifest(root / 'evaluation/manifests/real_sample.json').clips) == 5
    assert len(load_manifest(root / 'evaluation/manifests/proposed_sample.json').clips) == 20


def test_planned_clip_requires_a_filename_for_access(tmp_path):
    row = metadata(); row.pop('filename')
    with pytest.raises(ValueError, match='filename'):
        list(clips(Manifest(clips=[row]), tmp_path))
