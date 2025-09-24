from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from retriever import SessionState


@dataclass
class SessionRegistry:
    sessions: Dict[str, SessionState] = field(default_factory=dict)

    def get(self, session_id: Optional[str]) -> SessionState:
        if not session_id:
            return SessionState()
        return self.sessions.setdefault(session_id, SessionState())

    def reset(self, session_id: Optional[str]) -> SessionState:
        state = SessionState()
        if session_id:
            self.sessions[session_id] = state
        return state


registry = SessionRegistry()
