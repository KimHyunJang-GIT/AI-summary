from __future__ import annotations

import logging

from backend.api.app_factory import create_app
from backend.api.settings import Settings
from backend.retriever.provider import real_retriever_factory

logger = logging.getLogger(__name__)
settings = Settings()
app = create_app(settings=settings, retriever_provider=lambda: real_retriever_factory(settings))
