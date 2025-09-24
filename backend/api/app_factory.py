from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request

from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ReindexRequest,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
    SessionResetResponse,
    SessionSummary,
)
from .llm_service import LLMService
from .session import registry

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings,
    retriever_provider: Callable[[], object],
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = None
        if settings.STARTUP_LOAD:
            try:
                retriever = retriever_provider()
                if hasattr(retriever, "ready"):
                    retriever.ready(wait=False)
                app.state.retriever = retriever
                logger.info("retriever initialised during startup")
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("failed to initialise retriever: %s", exc)
        yield
        # cleanup placeholder

    app = FastAPI(title="AI Summary Retriever", lifespan=lifespan)
    llm_service = LLMService.from_settings(settings)

    def _ensure_retriever(request: Request):
        retriever = request.app.state.retriever
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialised")
        return retriever

    def _build_session_summary(state) -> SessionSummary:
        recent = list(getattr(state, "recent_queries", []))[-10:]
        preferred_exts = [
            ext
            for ext, weight in sorted(
                state.preferred_exts.items(), key=lambda item: item[1], reverse=True
            )
            if weight > 0
        ][:5]
        owner_prior = [
            owner
            for owner, weight in sorted(
                state.owner_prior.items(), key=lambda item: item[1], reverse=True
            )
            if weight > 0
        ][:5]
        return SessionSummary(
            recent_queries=recent,
            preferred_exts=preferred_exts,
            owner_prior=owner_prior,
        )

    @app.post("/api/search", response_model=SearchResponse)
    def search(req: SearchRequest, retr=Depends(_ensure_retriever)) -> SearchResponse:
        session_id, session_state = registry.get_or_create(req.session_id)
        prior_history = session_state.get_chat_history()
        hits = retr.search(req.query, top_k=req.top_k, session=session_state)
        answer, used_llm, llm_error = llm_service.generate_reply(
            user_message=req.query,
            conversation=prior_history,
            hits=hits,
        )
        session_state.record_user_message(req.query)
        if answer:
            session_state.record_assistant_message(answer)
        summary = _build_session_summary(session_state)
        history_payload = [
            {"role": role, "text": text} for role, text in session_state.get_chat_history()
        ]
        answer_source = "llm" if used_llm and answer else "fallback" if answer else "none"
        return SearchResponse(
            session_id=session_id,
            results=hits,
            explain=[hit.get("match_reasons", []) for hit in hits],
            session=summary,
            answer=answer,
            answer_source=answer_source,
            history=history_payload,
            llm_error=llm_error,
        )

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest) -> FeedbackResponse:
        session_id, session_state = registry.get_or_create(req.session_id)
        action = (req.action or "").lower().strip()
        if action not in {"click", "pin", "like", "dislike"}:
            raise HTTPException(status_code=400, detail=f"unsupported action {req.action}")
        if action in {"click", "pin"} and req.doc_id is None:
            raise HTTPException(status_code=400, detail="doc_id is required for click/pin actions")
        if action in {"like", "dislike"} and not (req.ext or req.owner):
            raise HTTPException(status_code=400, detail="like/dislike requires ext or owner metadata")
        if action == "click":
            session_state.record_click(doc_id=req.doc_id, ext=req.ext, owner=req.owner)
        elif action == "pin":
            session_state.record_pin(doc_id=req.doc_id, ext=req.ext, owner=req.owner)
        elif action == "like":
            session_state.record_like(ext=req.ext, owner=req.owner)
        elif action == "dislike":
            session_state.record_dislike(ext=req.ext, owner=req.owner)
        summary = _build_session_summary(session_state)
        return FeedbackResponse(session_id=session_id, status="ok", session=summary)

    @app.post("/api/session/reset", response_model=SessionResetResponse)
    def reset_session(req: FeedbackRequest) -> SessionResetResponse:
        session_id, state = registry.reset(req.session_id)
        summary = _build_session_summary(state)
        return SessionResetResponse(
            session_id=session_id,
            recent_queries=summary.recent_queries,
            history=[],
        )

    @app.post("/api/reindex", response_model=ReindexResponse)
    def reindex(req: ReindexRequest, retr=Depends(_ensure_retriever)) -> ReindexResponse:
        retr.ready(rebuild=req.force, wait=False)
        return ReindexResponse(status="scheduled")

    return app
