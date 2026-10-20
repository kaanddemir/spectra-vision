from datetime import date
import json
import subprocess
from pathlib import Path

import pytest
from evaluation.outputs import new_report_dir, write_prediction


def test_output_layout_is_separate_and_exclusive(tmp_path):
    key = 'a' * 64
    path = write_prediction(tmp_path / 'predictions', 'new-data', key, 'clip', {'analysis_key': key})
    assert path == tmp_path / 'predictions' / 'new-data' / key / 'clip.json'
    assert json.loads(path.read_text())['analysis_key'] == key
    with pytest.raises(FileExistsError):
        write_prediction(tmp_path / 'predictions', 'new-data', key, 'clip', {})
    report = new_report_dir(tmp_path / 'reports', 'run', date(2026, 10, 18))
    assert report.name == '2026-10-18_run'
    with pytest.raises(FileExistsError):
        new_report_dir(tmp_path / 'reports', 'run', date(2026, 10, 18))


@pytest.mark.parametrize('dataset,key,clip', [('../source', 'a'*64, 'clip'), ('data', 'bad', 'clip'),
                                              ('data', 'a'*64, '../clip')])
def test_invalid_output_paths_are_refused(tmp_path, dataset, key, clip):
    with pytest.raises(ValueError):
        write_prediction(tmp_path, dataset, key, clip, {})
    assert not list(tmp_path.iterdir())


def test_generated_output_roots_are_ignored():
    root = Path(__file__).resolve().parents[2]
    for name in ['evaluation/predictions/source/key/clip.json', 'evaluation/reports/date_run/summary.md']:
        assert subprocess.run(['git', 'check-ignore', '--quiet', name], cwd=root).returncode == 0


def test_prediction_symlink_escape_and_key_mismatch_are_refused(tmp_path):
    root = tmp_path / 'predictions'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / 'data').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        write_prediction(root, 'data', 'a'*64, 'clip', {})
    with pytest.raises(ValueError, match='does not match'):
        write_prediction(root, 'other', 'a'*64, 'clip', {'analysis_key': 'b'*64})
    assert not list(outside.iterdir())
