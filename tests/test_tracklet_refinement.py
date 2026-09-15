from __future__ import annotations

import pytest

from football_tracking.tracklet_refinement import TrackletRefinementError, refine_tracklets, tracklet_conflict_report
from football_tracking.tracking import TrackObservation


def rows(tracklet: str = "shot-0:t1") -> list[TrackObservation]:
    return [TrackObservation(tracklet, frame, frame, (0, 0, 10, 10), 0.9) for frame in range(6)]


def test_refinement_keeps_raw_rows_and_applies_reviewed_nonoverlapping_splits() -> None:
    original = rows()
    segments = refine_tracklets(original, {"shot-0:t1": [{"segment_id": "a", "start_frame": 0, "end_frame": 3}, {"segment_id": "b", "start_frame": 3, "end_frame": 6}]})
    assert [segment.segment_id for segment in segments] == ["a", "b"]
    assert [row.frame_index for row in original] == list(range(6))
    assert [row.frame_index for row in segments[0].observations] == [0, 1, 2]


def test_refinement_rejects_overlapping_or_empty_splits() -> None:
    with pytest.raises(TrackletRefinementError, match="overlap"):
        refine_tracklets(rows(), {"shot-0:t1": [{"start_frame": 0, "end_frame": 4}, {"start_frame": 3, "end_frame": 6}]})
    with pytest.raises(TrackletRefinementError):
        refine_tracklets(rows(), {"shot-0:t1": [{"start_frame": -1, "end_frame": 3, "segment_id": "invalid"}]})


def test_refinement_rejects_duplicate_segment_ids() -> None:
    with pytest.raises(TrackletRefinementError, match="globally unique"):
        refine_tracklets(rows("shot-0:t1") + rows("shot-0:t2"), {"shot-0:t1": [{"start_frame": 0, "end_frame": 6, "segment_id": "same"}], "shot-0:t2": [{"start_frame": 0, "end_frame": 6, "segment_id": "same"}]})


def test_refinement_rejects_splits_for_unknown_tracklets() -> None:
    with pytest.raises(TrackletRefinementError, match="unknown tracklets"):
        refine_tracklets(rows(), {"shot-0:missing": [{"start_frame": 0, "end_frame": 1}]})


def test_tracklet_conflict_report_is_a_review_queue() -> None:
    conflict_rows = [TrackObservation("shot-0:a", 0, 0, (0, 0, 10, 10), 0.9), TrackObservation("shot-0:b", 0, 0, (1, 1, 11, 11), 0.9)]
    report = tracklet_conflict_report(conflict_rows)
    assert report["status"] == "review_required"
    assert report["conflict_count"] == 1
    assert report["suspect_tracklets"] == ["shot-0:a", "shot-0:b"]
