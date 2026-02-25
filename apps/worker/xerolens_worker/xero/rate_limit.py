from __future__ import annotations

import time


def throttle(seconds: float = 0.2) -> None:
    time.sleep(seconds)
