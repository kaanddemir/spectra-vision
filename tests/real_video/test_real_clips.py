"""Explicit opt-in integration only; never run as part of the default suite."""
from pathlib import Path
import pytest

pytestmark = pytest.mark.real_video


def test_real_clip_evidence_cache_only(tmp_path):
    from scripts.check_real_clips import main
    repo = Path(__file__).resolve().parents[2]
    assert main(['--manifest', str(repo / 'evaluation/manifests/real_sample.json'),
                 '--clips', 'bolide', '--cache-only', '--out', str(tmp_path / 'evidence-check')]) == 0
