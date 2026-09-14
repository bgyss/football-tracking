from __future__ import annotations

import pytest

from football_tracking.memory import MemoryBudget, MemoryBudgetExceeded


def test_memory_budget_records_peak_and_rejects_an_over_budget_stage() -> None:
    readings = iter([128.0, 257.0])
    budget = MemoryBudget(limit_mb=256.0, read_rss_mb=lambda: next(readings))

    assert budget.sample("detector") == pytest.approx(128.0)
    with pytest.raises(MemoryBudgetExceeded, match="tracking"):
        budget.sample("tracking")

    assert budget.peak_mb == pytest.approx(257.0)
    assert budget.samples == {"detector": 128.0, "tracking": 257.0}
