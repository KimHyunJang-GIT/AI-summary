from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Settings:
    TESTING: bool = False
    STARTUP_LOAD: bool = True

    def __init__(
        self,
        TESTING: bool | None = None,
        STARTUP_LOAD: bool | None = None,
    ) -> None:
        if TESTING is not None:
            self.TESTING = TESTING
        else:
            self.TESTING = os.getenv("APP_TESTING", "0") == "1"
        if STARTUP_LOAD is not None:
            self.STARTUP_LOAD = STARTUP_LOAD
        else:
            self.STARTUP_LOAD = os.getenv("APP_STARTUP_LOAD", "1") != "0"
