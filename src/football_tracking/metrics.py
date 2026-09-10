"""Instrumentation and benchmark summaries that do not imply ground truth."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tracking import TrackObservation


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def package_version(name: str, fallback: str = "unavailable") -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return fallback


def system_info() -> dict[str, str]:
    return {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}


def summarize_tracks(rows: Iterable[TrackObservation]) -> dict[str, Any]:
    materialized = list(rows)
    by_frame: dict[int, list[str]] = {}
    by_track: dict[str, list[int]] = {}
    for row in materialized:
        by_frame.setdefault(row.frame_index, []).append(row.tracklet_id)
        by_track.setdefault(row.tracklet_id, []).append(row.frame_index)
    duplicate_ids = sum(len(ids) - len(set(ids)) for ids in by_frame.values())
    return {
        "observation_count": len(materialized),
        "unique_tracklets": len(by_track),
        "multi_frame_tracklets": sum(len(set(frames)) > 1 for frames in by_track.values()),
        "duplicate_track_ids_within_frame": duplicate_ids,
        "first_frame": min((row.frame_index for row in materialized), default=None),
        "last_frame": max((row.frame_index for row in materialized), default=None),
    }


class WallTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started
