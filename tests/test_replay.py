from __future__ import annotations

import json

import pytest

from football_tracking.replay import (
    PlayAlignment,
    PlayAnchor,
    ReplayAlignmentError,
    load_play_alignment,
)


def test_play_time_is_anchor_relative_and_shot_scoped() -> None:
    alignment = PlayAlignment("play-1", (PlayAnchor("shot-0", 120, "snap"), PlayAnchor("shot-1", 820, "snap")))

    assert alignment.shots() == ("shot-0", "shot-1")
    assert alignment.play_time_s("shot-0", 120, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-1", 820, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-0", 150, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-1", 850, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-2", 10, 60.0) is None


def test_alignment_requires_review_two_shots_and_positive_fps(tmp_path) -> None:
    unreviewed = tmp_path / "unreviewed.json"
    unreviewed.write_text(json.dumps({"play_id": "play-1", "anchors": []}), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(unreviewed)

    single = tmp_path / "single.json"
    single.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [{"shot_id": "shot-0", "source_frame": 120, "event": "snap"}],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(single)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-0", "source_frame": 130, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(duplicate)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    alignment = load_play_alignment(good)
    assert alignment.play_id == "play-1"
    assert alignment.shots() == ("shot-0", "shot-1")

    with pytest.raises(ValueError):
        alignment.play_time_s("shot-0", 120, 0.0)
