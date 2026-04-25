"""Cooperative spot-preemption / Ctrl-C handling.

GCE/TPU sends SIGTERM ~30 s before reclaiming a spot slice. We trap it and
flip a flag so the training loop can write a final checkpoint and exit
cleanly. The launch script's outer `while` retries the run from the fresh
checkpoint.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass


@dataclass
class StopState:
    requested: bool = False
    signum: int | None = None


def install(signums: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)) -> StopState:
    state = StopState()

    def _handler(signum: int, _frame) -> None:  # noqa: ANN001
        state.requested = True
        state.signum = signum

    for s in signums:
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            # Not the main thread; XLA spawn workers can ignore.
            pass
    return state
