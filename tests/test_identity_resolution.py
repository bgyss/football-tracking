from __future__ import annotations

from football_tracking.identity import TeamEvidence
from football_tracking.identity_resolution import ResolutionPolicy, resolve_play_identities
from football_tracking.replay import FieldTrack


def track(tracklet_id: str, shot_id: str, x: float) -> FieldTrack:
    return FieldTrack(tracklet_id, shot_id, tuple((index * 0.1, x + index * 0.1, 20.0) for index in range(8)))


def test_resolver_handles_three_views_and_reports_all_shots() -> None:
    tracks = {
        "shot-0": [track("shot-0:a", "shot-0", 10.0)],
        "shot-1": [track("shot-1:a", "shot-1", 10.0)],
        "shot-2": [track("shot-2:a", "shot-2", 10.0)],
    }
    teams = {key: TeamEvidence("DET", 0.95, "reviewed", 8, 1.0, 0.0) for key in ("shot-0:a", "shot-1:a", "shot-2:a")}
    result = resolve_play_identities({"play_id": "p1", "shot_ids": tuple(tracks)}, tracks, teams=teams)
    assert result.status == "resolved"
    assert len(result.accepted_links) == 3
    assert result.unresolved_shots == ()
    assert result.candidate_evidence
    assert all("paired_sample_count" in record for record in result.candidate_evidence)


def test_resolver_marks_missing_middle_view_partial() -> None:
    tracks = {"shot-0": [track("shot-0:a", "shot-0", 10.0)], "shot-1": [], "shot-2": [track("shot-2:a", "shot-2", 10.0)]}
    teams = {"shot-0:a": TeamEvidence("DET", 0.95, "reviewed", 8), "shot-2:a": TeamEvidence("DET", 0.95, "reviewed", 8)}
    result = resolve_play_identities({"play_id": "p1", "shot_ids": ("shot-0", "shot-1", "shot-2")}, tracks, teams=teams)
    assert result.status == "partially_resolved"
    assert result.unresolved_shots == ("shot-1",)
    assert result.unmatched_tracklets == ()


def test_policy_is_explicit() -> None:
    assert ResolutionPolicy(threshold=0.8).threshold == 0.8


def test_resolver_reports_unmatched_segments_as_partial() -> None:
    tracks = {
        "shot-0": [track("shot-0:a", "shot-0", 10.0), track("shot-0:b", "shot-0", 40.0)],
        "shot-1": [track("shot-1:a", "shot-1", 10.0)],
    }
    teams = {key: TeamEvidence("DET", 0.95, "reviewed", 8) for key in ("shot-0:a", "shot-0:b", "shot-1:a")}
    frames = {"shot-0:a": tuple(range(8)), "shot-0:b": tuple(range(8, 16)), "shot-1:a": tuple(range(8))}
    result = resolve_play_identities({"play_id": "p1", "shot_ids": ("shot-0", "shot-1")}, tracks, teams=teams, tracklet_frames=frames)
    assert result.status == "partially_resolved"
    assert result.unmatched_tracklets == ("shot-0:b",)
