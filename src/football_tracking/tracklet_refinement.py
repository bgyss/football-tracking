"""Reviewed split overlays and conservative local identity segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .tracking import TrackObservation


class TrackletRefinementError(ValueError):
    """Raised when a reviewed tracklet split is inconsistent."""


@dataclass(frozen=True, slots=True)
class IdentitySegment:
    segment_id: str
    source_tracklet_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    observations: tuple[TrackObservation, ...]

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise TrackletRefinementError("identity segment interval is invalid")
        if any(row.tracklet_id != self.source_tracklet_id or not self.start_frame <= row.frame_index < self.end_frame for row in self.observations):
            raise TrackletRefinementError("segment observations do not fit its source interval")


def refine_tracklets(
    rows: Sequence[TrackObservation],
    reviewed_splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[IdentitySegment, ...]:
    """Build immutable local segments without mutating raw tracker observations.

    A split overlay is accepted only when it is explicitly supplied by review.
    The default creates one segment per source tracklet. Intervals in one source
    tracklet may be disjoint, but they may not overlap.
    """

    by_track: dict[str, list[TrackObservation]] = {}
    for row in rows:
        by_track.setdefault(row.tracklet_id, []).append(row)
    result: list[IdentitySegment] = []
    for tracklet_id in sorted(by_track):
        values = sorted(by_track[tracklet_id], key=lambda row: (row.frame_index, row.pts))
        shot_id = tracklet_id.split(":", 1)[0]
        specs = list((reviewed_splits or {}).get(tracklet_id, ()))
        if not specs:
            specs = [{"segment_id": tracklet_id, "start_frame": values[0].frame_index, "end_frame": values[-1].frame_index + 1}]
        parsed: list[tuple[int, int, str]] = []
        for index, spec in enumerate(specs):
            try:
                start, end = int(spec["start_frame"]), int(spec["end_frame"])
            except (KeyError, TypeError, ValueError) as error:
                raise TrackletRefinementError(f"invalid split for {tracklet_id}") from error
            if end <= start or start < values[0].frame_index or end > values[-1].frame_index + 1:
                raise TrackletRefinementError(f"split interval is outside {tracklet_id}")
            parsed.append((start, end, str(spec.get("segment_id", f"{tracklet_id}:segment-{index}"))))
        parsed.sort()
        for previous, current in zip(parsed, parsed[1:]):
            if current[0] < previous[1]:
                raise TrackletRefinementError(f"split intervals overlap for {tracklet_id}")
        for start, end, segment_id in parsed:
            selected = tuple(row for row in values if start <= row.frame_index < end)
            if not selected:
                raise TrackletRefinementError(f"split {segment_id} contains no observations")
            result.append(IdentitySegment(segment_id, tracklet_id, shot_id, start, end, selected))
        covered = {row.frame_index for segment in result if segment.source_tracklet_id == tracklet_id for row in segment.observations}
        if covered != {row.frame_index for row in values}:
            raise TrackletRefinementError(f"reviewed splits do not cover every observed frame for {tracklet_id}")
    return tuple(sorted(result, key=lambda segment: segment.segment_id))
