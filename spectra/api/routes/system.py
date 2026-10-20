"""Server health."""
from __future__ import annotations

from fastapi import APIRouter, Request

from spectra.api import API_VERSION
from spectra.api.schemas import Health
from spectra.api.system import analysis_readiness, database_health, memory_health

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Report database, model-registry and memory status. Always 200; readiness is in the body."""
    state = request.app.state
    database = database_health(state.database)
    models = state.models.status()
    memory = memory_health(state.pressure())
    analysis = analysis_readiness(database, models, memory)
    return Health(status="ok" if analysis.available else "degraded", version=API_VERSION,
                  database=database, models=models, memory=memory, analysis=analysis,
                  library_folders=len(state.library))

