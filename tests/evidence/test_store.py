import json

import pytest

from spectra.evidence.build import build
from spectra.evidence.contracts import ReviewConfig
from spectra.evidence.store import STAGING, EvidenceStore
from tests.evidence import synth as S


@pytest.fixture
def built():
    return build(S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.3), S.cut_in(9.0, 9.5)]))


def test_round_trip_verifies_and_reads_packages_on_request(tmp_path, built):
    timeline, packages = built
    store = EvidenceStore(tmp_path)
    d = store.write(timeline, packages)
    assert d.name == timeline.build_key and not (tmp_path / STAGING).exists() or not any((tmp_path / STAGING).iterdir())
    found = store.load(timeline.build_key, timeline.analysis_key)
    assert found.timeline == timeline and found.reason is None
    for p in packages:
        assert store.package(timeline.build_key, p.proposal.event_id) == p


def test_repeated_writes_produce_identical_bytes(tmp_path, built):
    a, b = EvidenceStore(tmp_path / "a"), EvidenceStore(tmp_path / "b")
    da, db = a.write(*built), b.write(*build(S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.3),
                                                       S.cut_in(9.0, 9.5)])))
    files = sorted(p.relative_to(da) for p in da.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(db) for p in db.rglob("*") if p.is_file())
    assert all((da / f).read_bytes() == (db / f).read_bytes() for f in files)
    mtime = (da / "timeline.json").stat().st_mtime_ns
    a.write(*built)  # a valid build is kept, not rewritten
    assert (da / "timeline.json").stat().st_mtime_ns == mtime


def test_a_different_configuration_is_a_different_build(tmp_path, built):
    store = EvidenceStore(tmp_path)
    first = store.write(*built)
    second = store.write(*build(S.result([S.approach(5.0, 7.0)]), ReviewConfig(pre_s=5.0)))
    assert first != second and first.exists() and second.exists()


def test_a_build_is_not_returned_for_another_analysis(tmp_path, built):
    store = EvidenceStore(tmp_path)
    store.write(*built)
    found = store.load(built[0].build_key, "c" * 64)
    assert found.timeline is None and "different analysis" in found.reason


def test_tampered_files_are_rejected_with_a_reason(tmp_path, built):
    timeline, packages = built
    store = EvidenceStore(tmp_path)
    d = store.write(timeline, packages)
    path = d / "packages" / f"{packages[0].proposal.event_id}.json"
    path.write_text(path.read_text().replace("supported", "uncertain", 1))
    found = store.load(timeline.build_key)
    assert found.timeline is None and "does not match its recorded hash" in found.reason
    with pytest.raises(ValueError):
        store.package(timeline.build_key, packages[0].proposal.event_id)


def test_a_rejected_build_is_set_aside_not_deleted_when_rewritten(tmp_path, built):
    timeline, packages = built
    store = EvidenceStore(tmp_path)
    d = store.write(timeline, packages)
    (d / "timeline.json").write_text("{}")
    store.write(timeline, packages)
    assert store.load(timeline.build_key).timeline == timeline
    assert len(list(tmp_path.glob(f"{timeline.build_key}.rejected.*"))) == 1


def test_an_incompatible_evidence_version_is_rejected(tmp_path, built):
    store = EvidenceStore(tmp_path)
    d = store.write(*built)
    manifest = json.loads((d / "manifest.json").read_text())
    (d / "manifest.json").write_text(json.dumps({**manifest, "evidence_version": "0"}))
    found = store.load(built[0].build_key)
    assert found.timeline is None and "incompatible evidence version" in found.reason


def test_only_matching_packages_are_stored(tmp_path, built):
    timeline, packages = built
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path).write(timeline, packages[:-1])
