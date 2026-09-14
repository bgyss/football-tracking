"""Small dependency-free host-memory guard for local tracking runs."""

from __future__ import annotations

import resource
import sys
from collections.abc import Callable


class MemoryBudgetExceeded(RuntimeError):
    """Raised after a stage pushes the process above its declared host-RSS budget."""


def process_peak_rss_mb() -> float:
    """Return the process high-water RSS in MiB on macOS and Unix-like hosts."""

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSD-like CI environments report KiB.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


class MemoryBudget:
    """Samples process peak RSS and stops a run that has exceeded its budget."""

    def __init__(self, limit_mb: float | None, read_rss_mb: Callable[[], float] = process_peak_rss_mb) -> None:
        if limit_mb is not None and limit_mb <= 0:
            raise ValueError("memory budget must be positive when configured")
        self.limit_mb = limit_mb
        self._read_rss_mb = read_rss_mb
        self.peak_mb = 0.0
        self.samples: dict[str, float] = {}
        self.hard_limit_applied = False

    def enforce_process_limit(self) -> None:
        """Apply an address-space ceiling before heavyweight model construction."""

        if self.limit_mb is None:
            return
        limit_bytes = int(self.limit_mb * 1024 * 1024)
        try:
            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
            if current_hard != resource.RLIM_INFINITY and current_hard < limit_bytes:
                limit_bytes = current_hard
            if current_soft == resource.RLIM_INFINITY or current_soft > limit_bytes:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            self.hard_limit_applied = True
        except (OSError, ValueError):
            # RSS sampling remains active on platforms that do not permit a
            # process address-space limit; the run still fails after a sample.
            self.hard_limit_applied = False

    def sample(self, stage: str) -> float:
        value = float(self._read_rss_mb())
        self.peak_mb = max(self.peak_mb, value)
        self.samples[str(stage)] = value
        if self.limit_mb is not None and value > self.limit_mb:
            raise MemoryBudgetExceeded(
                f"{stage} reached {value:.1f} MiB, above --max-memory-mb {self.limit_mb:.1f}; "
                "reduce model/mask size or raise the explicit budget"
            )
        return value
