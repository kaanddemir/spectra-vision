"""Human review decisions and edits on the event proposals of an analysis."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from spectra.api.edits import EditConflict, EditInvalid
from spectra.api.evidence import EvidenceNotFound, EvidenceUnavailable
from spectra.api.reviews import ReviewConflict, ReviewNotFound, ReviewRefused
from spectra.api.schemas import (EventEdit, EventEdits, EventEditUpdate, ReviewDecision, ReviewDecisions,
                                 ReviewDecisionUpdate)

router = APIRouter(prefix="/analyses/{analysis_id}", tags=["reviews"])
AnalysisId = Path(pattern=r"^[0-9a-f]{32}$")
EventId = Path(pattern=r"^[0-9a-f]{16}$")


@router.get("/reviews", response_model=ReviewDecisions, responses={404: {"description": "analysis not found"}})
def list_decisions(request: Request, analysis_id: str = AnalysisId) -> ReviewDecisions:
    """Recorded decisions; a proposal without one is not reviewed. An analysis without proposals has none."""
    try:
        return request.app.state.reviews.list(analysis_id)
    except ReviewNotFound as exc:
        raise HTTPException(404, str(exc)) from None


@router.put("/reviews/{event_id}", response_model=ReviewDecision,
            responses={404: {"description": "analysis or proposal not found"},
                       409: {"description": "no proposals to review, or the decision changed since "
                                            "`expected_revision`; `detail.current` holds the latest decision"}})
def set_decision(body: ReviewDecisionUpdate, request: Request, analysis_id: str = AnalysisId,
                 event_id: str = EventId) -> ReviewDecision:
    """Accept or reject a proposal, or return it to not reviewed. The analysis and its evidence are unchanged."""
    try:
        return request.app.state.reviews.set(analysis_id, event_id, body.decision, body.expected_revision)
    except ReviewNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except ReviewRefused as exc:
        raise HTTPException(409, str(exc)) from None
    except ReviewConflict as exc:
        raise HTTPException(409, {"message": str(exc), "current": exc.current.model_dump(mode="json")}) from None


@router.get("/edits", response_model=EventEdits, responses={404: {"description": "analysis not found"}})
def list_edits(request: Request, analysis_id: str = AnalysisId) -> EventEdits:
    """Recorded edits; a proposal without one keeps its proposed times and has no description."""
    try:
        return request.app.state.edits.list(analysis_id)
    except ReviewNotFound as exc:
        raise HTTPException(404, str(exc)) from None


@router.put("/edits/{event_id}", response_model=EventEdit,
            responses={404: {"description": "analysis or proposal not found"},
                       409: {"description": "no proposals, video duration unavailable, or the edit changed since "
                                            "`expected_revision`; `detail.current` holds the latest edit"},
                       422: {"description": "malformed, or times out of order or outside the video"}})
def set_edit(body: EventEditUpdate, request: Request, analysis_id: str = AnalysisId,
             event_id: str = EventId) -> EventEdit:
    """Replace a proposal's adjusted times and description. The proposal, its evidence and its review
    decision are unchanged; null times and description restore the proposal as detected."""
    try:
        return request.app.state.edits.set(analysis_id, event_id, body.t_start, body.t_end, body.description,
                                           body.expected_revision)
    except (ReviewNotFound, EvidenceNotFound) as exc:
        raise HTTPException(404, str(exc) or "not found") from None
    except ReviewRefused as exc:
        raise HTTPException(409, str(exc)) from None
    except EvidenceUnavailable as exc:
        raise HTTPException(409, f"the video's duration is not available to check the times: {exc}") from None
    except EditInvalid as exc:
        raise HTTPException(422, {"message": str(exc)}) from None
    except EditConflict as exc:
        raise HTTPException(409, {"message": str(exc), "current": exc.current.model_dump(mode="json")}) from None
