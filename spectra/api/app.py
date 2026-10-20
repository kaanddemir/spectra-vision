"""Application factory. Startup creates the data directories, initializes the database schema and resumes
interrupted media probes and settles interrupted analyses; shutdown cancels the running analysis, waits for
its worker and kills a running probe helper."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI

from spectra.api import API_VERSION
from spectra.api.evidence import EvidenceService
from spectra.api.frontend import is_build, mount_frontend
from spectra.api.jobs import AnalysisService, JobFactory, WorkerRunner, standard_job
from spectra.api.library import LibraryFolders
from spectra.api.picker import FolderPicker, choose_folder
from spectra.api.probe import ProbeRunner
from spectra.api.edits import EditService
from spectra.api.export import ExportService
from spectra.api.reviews import ReviewService
from spectra.api.routes import analyses, evidence, library, reviews, system, videos
from spectra.api.security import LocalRequestGuard
from spectra.api.settings import LOCAL_HOSTNAMES, Settings
from spectra.api.store import Database
from spectra.api.system import ModelRegistryCheck, PressureReader
from spectra.api.videos import ProbeFunction, VideoRegistry
from spectra.engine.resources import memory_pressure_level


def create_app(settings: Optional[Settings] = None, *,
               pressure: PressureReader = memory_pressure_level,
               probe: Optional[ProbeFunction] = None, job_factory: Optional[JobFactory] = None,
               run_worker: Optional[WorkerRunner] = None,
               pick_folder: FolderPicker = choose_folder) -> FastAPI:
    """``probe``, ``job_factory``, ``run_worker`` and ``pick_folder`` replace the media-probe helper, the
    analysis profile, the worker launcher and the native folder dialog in tests."""
    settings = settings or Settings()
    database = Database(settings.database_path)
    runner = ProbeRunner()
    folders = LibraryFolders(database, settings.library_dirs, settings.data_dir)
    registry = VideoRegistry(database, folders, settings.data_dir, probe or runner.run)
    models = ModelRegistryCheck(settings.models_dir)
    evidence_service = EvidenceService(database, settings.cache_dir, settings.evidence_dir)
    jobs = AnalysisService(database, registry, models, pressure,
                           job_factory or standard_job(settings.models_dir, settings.cache_dir),
                           on_result=evidence_service.build_from_outcome,
                           **({"run": run_worker} if run_worker else {}))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.prepare()
        database.initialize()
        registry.start()
        evidence_service.start()
        jobs.start()
        try:
            yield
        finally:
            evidence_service.close()
            jobs.close()
            runner.close()
            registry.close()

    app = FastAPI(title="Spectra", version=API_VERSION, lifespan=lifespan,
                  openapi_url="/api/openapi.json", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.database = database
    app.state.models = models
    app.state.pressure = pressure
    app.state.library = folders
    app.state.pick_folder = pick_folder
    app.state.videos = registry
    app.state.analyses = jobs
    app.state.evidence = evidence_service
    app.state.reviews = ReviewService(database)
    app.state.edits = EditService(database, evidence_service.duration)
    app.state.export = ExportService(database, evidence_service.store)
    for module in (system, library, videos, analyses, evidence, reviews):
        app.include_router(module.router, prefix="/api")
    if settings.frontend_dir is not None and is_build(settings.frontend_dir):
        mount_frontend(app, settings.frontend_dir)
    app.add_middleware(LocalRequestGuard, hostnames=LOCAL_HOSTNAMES, origins=settings.allowed_origins)
    return app
