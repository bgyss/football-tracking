from __future__ import annotations

import cv2
import numpy as np
import pytest

import football_tracking.video as video_module
from football_tracking.video import (
    ShotBoundary,
    VideoInfo,
    detect_shots,
    iter_video_frames,
    shot_ranges,
)


def test_manual_cut_is_deduplicated_and_scene_candidates_are_ordered() -> None:
    frames = [np.full((16, 16, 3), 20, dtype=np.uint8) for _ in range(5)]
    frames.extend(np.full((16, 16, 3), 220, dtype=np.uint8) for _ in range(5))

    boundaries = detect_shots(frames, fps=10.0, manual_boundaries=[5, 5])

    assert [boundary.frame_index for boundary in boundaries] == [0, 5]
    assert boundaries[0].reason == "start"
    assert boundaries[1].reason == "manual"
    assert shot_ranges(10, boundaries) == [(0, 5), (5, 10)]


def test_shot_boundary_rejects_negative_frame() -> None:
    try:
        ShotBoundary(frame_index=-1, pts=0, reason="bad", confidence=0.0)
    except ValueError as error:
        assert "frame_index" in str(error)
    else:
        raise AssertionError("negative boundary should fail")


def test_video_info_and_streaming_frames_preserve_order(tmp_path) -> None:
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
    writer.release()

    info = VideoInfo.from_path(path)
    rows = list(iter_video_frames(path))

    assert info.width == 32
    assert info.height == 24
    assert info.source_fps > 0
    assert info.duration_s == pytest.approx(0.4, abs=0.02)
    assert len(rows) == 4
    assert [row[0] for row in rows] == [0, 1, 2, 3]
    assert [row[1] for row in rows] == sorted(row[1] for row in rows)
    assert all(frame.shape == (24, 32, 3) for _, _, frame in rows)


def test_frame_pts_are_cached_between_streaming_passes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cached.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    video_module._frame_pts_cached.cache_clear()
    calls = []
    original_run = video_module.subprocess.run

    def counted_run(command, *args, **kwargs):
        if "frame=best_effort_timestamp" in " ".join(str(part) for part in command):
            calls.append(command)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(video_module.subprocess, "run", counted_run)
    list(video_module.iter_video_frames(path))
    list(video_module.iter_video_frames(path))
    assert len(calls) == 1


def test_bounded_streaming_can_use_declared_pts_step_without_full_index(tmp_path) -> None:
    path = tmp_path / "bounded.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    rows = list(iter_video_frames(path, start_frame=1, end_frame=3, pts_per_frame=100.0))
    assert [(row[0], row[1]) for row in rows] == [(1, 100), (2, 200)]
