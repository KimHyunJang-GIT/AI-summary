from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from uuid import uuid4

from retriever import SessionState


@dataclass
class SessionRegistry:
    sessions: Dict[str, SessionState] = field(default_factory=dict)

    def get_or_create(self, session_id: Optional[str]) -> Tuple[str, SessionState]:
        if session_id:
            return session_id, self.sessions.setdefault(session_id, SessionState())

        new_id = uuid4().hex
        state = SessionState()
        self.sessions[new_id] = state
        return new_id, state

    def reset(self, session_id: Optional[str]) -> Tuple[str, SessionState]:
        new_state = SessionState()
        if not session_id:
            session_id = uuid4().hex
        self.sessions[session_id] = new_state
        return session_id, new_state


registry = SessionRegistry()
