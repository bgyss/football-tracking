"""Reviewed play-time alignment and cross-shot identity candidate scoring.

Shots are tracked independently, so nothing in image coordinates survives a cut.
This module converts media time into a shared play time anchored on a reviewed
event, and scores cross-shot tracklet pairs only from view-invariant evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ReplayAlignmentError(ValueError):
    """Raised when an alignment cannot support a cross-shot join."""


@dataclass(frozen=True, slots=True)
class PlayAnchor:
    shot_id: str
    source_frame: int
    event: str

    def __post_init__(self) -> None:
        if self.source_frame < 0:
            raise ValueError("anchor source_frame must be non-negative")
        if not self.event:
            raise ValueError("anchor event must be named")


@dataclass(frozen=True, slots=True)
class PlayAlignment:
    play_id: str
    anchors: tuple[PlayAnchor, ...]

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(anchor.shot_id for anchor in self.anchors))

    def play_time_s(self, shot_id: str, frame_index: int, fps: float) -> float | None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for anchor in self.anchors:
            if anchor.shot_id == shot_id:
                return (frame_index - anchor.source_frame) / fps
        return None


def load_play_alignment(path: str | Path) -> PlayAlignment:
    """Load reviewed snap anchors. Model-generated alignments are refused."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayAlignmentError(f"unable to read alignment: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise ReplayAlignmentError("alignment must be explicitly marked reviewed: true")
    play_id = str(value.get("play_id") or "")
    if not play_id:
        raise ReplayAlignmentError("alignment must name a play_id")
    raw_anchors = value.get("anchors")
    if not isinstance(raw_anchors, list) or len(raw_anchors) < 2:
        raise ReplayAlignmentError("alignment needs a reviewed anchor for at least two shots")
    anchors: list[PlayAnchor] = []
    seen: set[str] = set()
    for raw in raw_anchors:
        if not isinstance(raw, dict) or "shot_id" not in raw or "source_frame" not in raw:
            raise ReplayAlignmentError(f"invalid anchor: {raw!r}")
        shot_id = str(raw["shot_id"])
        if shot_id in seen:
            raise ReplayAlignmentError(f"duplicate anchor for {shot_id}")
        seen.add(shot_id)
        try:
            anchors.append(PlayAnchor(shot_id, int(raw["source_frame"]), str(raw.get("event", "snap"))))
        except (TypeError, ValueError) as error:
            raise ReplayAlignmentError(f"invalid anchor for {shot_id}: {error}") from error
    return PlayAlignment(play_id, tuple(sorted(anchors, key=lambda anchor: anchor.shot_id)))
