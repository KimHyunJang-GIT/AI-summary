from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None


class SessionSummary(BaseModel):
    recent_queries: List[str]
    preferred_exts: List[str]
    owner_prior: List[str]


class SearchResponse(BaseModel):
    session_id: Optional[str]
    results: List[dict]
    explain: List[List[str]]
    session: SessionSummary


class FeedbackRequest(BaseModel):
    session_id: Optional[str]
    doc_id: Optional[int] = None
    path: Optional[str] = None
    ext: Optional[str] = None
    owner: Optional[str] = None
    action: Literal["click", "pin", "like", "dislike"]


class FeedbackResponse(BaseModel):
    session_id: Optional[str]
    status: str
    session: Optional[SessionSummary] = None


class ReindexRequest(BaseModel):
    force: bool = False


class ReindexResponse(BaseModel):
    status: str


class SessionResetResponse(BaseModel):
    session_id: Optional[str]
    recent_queries: List[str]
