"""FastAPI service: shared fan power-consumption model across factories."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import db as dbmod
from .gear import (MAHALANOBIS_LIMIT, OperatingPoint, review_gear,
                   select_gear)
from .model import RidgeModel, train
from .schemas import (CorrectionUpdate, CreateVersionRequest, GearRequest,
                      GearReviewRequest, GearReviewResponse,
                      GearSelectionResponse, PredictRequest)
from .stats import InvalidStats, Stats, apply_correction

DB_PATH = os.environ.get("DB_PATH", os.path.join("data", "platform.db"))


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path if db_path is not None else DB_PATH
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    store = dbmod.Store(path)

    app = FastAPI(title="Fan Power Model Platform")
    app.state.store = store

    # ------------------------------------------------------------- helpers
    def _train_from_dicts(factories):
        """Apply each factory's correction and fit; may raise InvalidStats."""
        stats = [
            apply_correction(
                Stats.build(f["n"], f["sums"], f["outer"]), f["correction"]
            )
            for f in factories
        ]
        return train(stats)

    def _public(view: dict) -> dict:
        return {
            "version": view["version"],
            "lambda": view["lambda"],
            "lambda_scores": view["lambda_scores"],
            "factory_errors": view["factory_errors"],
            "corrections": {f["factory_id"]: f["correction"]
                            for f in view["factories"]},
            "model": view["model"],
            "created_at": view["created_at"],
        }

    def _gear_context(view: dict, factory_id: str):
        """One version's correction, summary and model for a factory."""
        for f in view["factories"]:
            if f["factory_id"] == factory_id:
                stats = Stats.build(f["n"], f["sums"], f["outer"])
                op = OperatingPoint.build(stats, f["correction"])
                model = RidgeModel.from_dict(view["model"])
                return f["correction"], op, model
        raise HTTPException(
            status_code=404, detail=f"unknown factory {factory_id!r}"
        )

    def _current_view():
        view = store.get_current()
        if view is None:
            raise HTTPException(status_code=503, detail="no model version yet")
        return view

    # ------------------------------------------------------------- routes
    @app.get("/health")
    def health():
        return {"status": "ok", "current_version": store.current_version_id()}

    @app.post("/versions", status_code=201)
    def create_version(req: CreateVersionRequest):
        """Onboard 3-8 factories and train a fresh model version."""
        try:
            result = _train_from_dicts([f.model_dump() for f in req.factories])
        except InvalidStats as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        factories = [f.model_dump() for f in req.factories]
        new_id = store.create_version(factories, result)
        return _public(store.get_version(new_id))

    @app.get("/versions/current")
    def get_current():
        view = store.get_current()
        if view is None:
            raise HTTPException(status_code=404, detail="no version yet")
        return _public(view)

    @app.get("/versions/{version_id}")
    def get_version(version_id: int):
        view = store.get_version(version_id)
        if view is None:
            raise HTTPException(status_code=404,
                                detail=f"version {version_id} not found")
        return _public(view)

    @app.post("/versions/{version_id}/correction")
    def update_correction(version_id: int, req: CorrectionUpdate):
        """Replace one factory's zero-point correction, based on a version.

        The correction, the retrained model and the per-factory errors take
        effect together as a new current version.  If the base version is
        not current (conflict) or recomputation fails, the current version
        is kept.
        """
        base = store.get_version(version_id)
        if base is None:
            raise HTTPException(status_code=404,
                                detail=f"version {version_id} not found")
        current = store.current_version_id()
        if current != version_id:
            raise HTTPException(
                status_code=409,
                detail=f"version conflict: based on {version_id}, "
                       f"current is {current}; current version kept",
            )
        factories = []
        found = False
        for f in base["factories"]:
            f = dict(f)
            if f["factory_id"] == req.factory_id:
                f["correction"] = req.correction
                found = True
            factories.append(f)
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"factory {req.factory_id!r} not in version {version_id}",
            )
        try:
            result = _train_from_dicts(factories)
        except InvalidStats as exc:
            # Recomputation failed: current version stays in effect.
            raise HTTPException(status_code=422,
                                detail=f"recomputation failed: {exc}")
        try:
            new_id = store.replace_correction(version_id, factories, result)
        except dbmod.Conflict as exc:
            raise HTTPException(status_code=409,
                                detail=f"{exc}; current version kept")
        return _public(store.get_version(new_id))

    @app.post("/predict")
    def predict(req: PredictRequest):
        """Predict per-minute power from raw speed/pressure readings.

        Always served from the committed current version (also while a
        recomputation is in flight); the version used is returned.
        """
        view = store.get_current()
        if view is None:
            raise HTTPException(status_code=503, detail="no model version yet")
        corr = {f["factory_id"]: f["correction"] for f in view["factories"]}
        if req.factory_id not in corr:
            raise HTTPException(
                status_code=404,
                detail=f"unknown factory {req.factory_id!r}",
            )
        model = RidgeModel.from_dict(view["model"])
        power = model.predict(req.speed, req.pressure + corr[req.factory_id])
        return {
            "factory_id": req.factory_id,
            "power": power,
            "version": view["version"],
        }

    @app.post("/gear/select", response_model=GearSelectionResponse)
    def gear_select(req: GearRequest):
        """Highest usable integer gear under pressure interval + power cap.

        Conditions are checked over the *whole* corrected pressure
        interval.  Everything -- correction, summary statistics and model
        -- comes from the one committed current version returned below.
        No gear is tried one by one: the feasible speed set is solved
        analytically.  Returns ``available: false`` when no gear qualifies.
        """
        view = _current_view()
        correction, op, model = _gear_context(view, req.factory_id)
        p_lo, p_hi = sorted((req.pressure_min, req.pressure_max))
        selection = select_gear(
            op, model,
            p_lo + correction, p_hi + correction,
            req.speed_min, req.speed_max, req.power_limit,
        )
        return {
            "factory_id": req.factory_id,
            "available": selection.available,
            "gear": selection.gear,
            "version": view["version"],
            "correction": correction,
            "power_limit": req.power_limit,
            "mahalanobis_limit": MAHALANOBIS_LIMIT,
            "max_power": selection.max_power,
            "max_distance": selection.max_distance,
        }

    @app.post("/gear/review", response_model=GearReviewResponse,
              responses={409: {"model": GearReviewResponse,
                               "description": "gear not usable under the "
                                              "current version; suggestion "
                                              "attached"}})
    def gear_review(req: GearReviewRequest):
        """Re-check a gear chosen earlier, against the current version.

        The original conditions are replayed.  If the version changed in
        between (e.g. a correction update), the *new* version's correction,
        summary and model decide.  An unusable gear is rejected with
        409 and the current suggestion is included.
        """
        view = _current_view()
        correction, op, model = _gear_context(view, req.factory_id)
        p_lo, p_hi = sorted((req.pressure_min, req.pressure_max))
        review = review_gear(
            op, model, req.gear,
            p_lo + correction, p_hi + correction,
            req.speed_min, req.speed_max, req.power_limit,
        )
        body = {
            "factory_id": req.factory_id,
            "accepted": review.accepted,
            "gear": req.gear,
            "version": view["version"],
            "correction": correction,
            "power_limit": req.power_limit,
            "mahalanobis_limit": MAHALANOBIS_LIMIT,
            "max_power": review.max_power,
            "max_distance": review.max_distance,
            "reasons": list(review.reasons),
        }
        if review.accepted:
            return body
        # Rejected: attach the current version's own suggestion.
        suggested = select_gear(
            op, model,
            p_lo + correction, p_hi + correction,
            req.speed_min, req.speed_max, req.power_limit,
        )
        body["suggestion"] = {
            "available": suggested.available,
            "gear": suggested.gear,
            "max_power": suggested.max_power,
            "max_distance": suggested.max_distance,
        }
        return JSONResponse(status_code=409, content=body)

    return app


# Run with:  uvicorn --factory app.main:create_app
# (factory pattern keeps `import app.main` free of side effects)
