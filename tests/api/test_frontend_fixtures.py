"""The web application's test fixtures are real API responses of the current contracts."""
from spectra.api.schemas import (Analysis, ClipResponse, Health, LibraryFolderInfo, LibraryListing, OverlayResponse,
                                 ReviewDecisions, Video, EventEdits)
from spectra.evidence.contracts import EvidencePackage, Timeline
from tests.api.frontend_fixtures import FIXTURES_PATH, build_fixtures, render


def test_fixtures_are_current():
    assert FIXTURES_PATH.read_text(encoding="utf-8") == render(), (
        "regenerate with: python -m tests.api.frontend_fixtures")


def test_fixtures_validate_against_the_response_models():
    fixtures = build_fixtures()
    for key in ("health", "healthDegraded"):
        Health.model_validate(fixtures[key])
    for key in ("folders", "twoFolders"):
        [LibraryFolderInfo.model_validate(f) for f in fixtures[key]]
    for key in ("listing", "sublisting"):
        LibraryListing.model_validate(fixtures[key])
    videos = {key: Video.model_validate(v) for key, v in fixtures["videos"].items()}
    [Analysis.model_validate(a) for a in fixtures["analyses"]]
    timeline = Timeline.model_validate(fixtures["timeline"])
    assert len(timeline.entries) == 3
    packages = {key: EvidencePackage.model_validate(p) for key, p in fixtures["packages"].items()}
    assert list(packages) == [e.event_id for e in timeline.entries]
    drawn = OverlayResponse.model_validate(fixtures["overlay"])
    assert drawn.overlay.event_id == timeline.entries[0].event_id and drawn.overlay.frames
    clip = ClipResponse.model_validate(fixtures["clip"])
    assert clip.clip.source_sha256 == packages[drawn.overlay.event_id].video.sha256
    assert videos["ready"].analysis.eligible and not videos["needs_check"].analysis.eligible
    reviews = ReviewDecisions.model_validate(fixtures["reviews"])
    assert reviews.analysis_id == fixtures["analyses"][0]["id"]
    assert {d.event_id for d in reviews.decisions} <= set(packages)
    edits = EventEdits.model_validate(fixtures["edits"])
    assert {e.event_id for e in edits.edits} <= set(packages) and edits.analysis_id == reviews.analysis_id
    assert videos["unsupported"].media_url is None and videos["probing"].media is None
