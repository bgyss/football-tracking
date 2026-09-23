"""Streaming video metadata, timestamps, and shot boundaries."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np


def _parse_ratio(value: str | int | float | None, default: float = 0.0) -> float:
    if value in (None, "", "N/A"):
        return default
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return default
    try:
        return float(text)
    except ValueError:
        return default


def _ffprobe(path: Path, *entries: str) -> dict | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        f"stream={','.join(entries)}:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=16)
def _frame_pts_cached(path_text: str, source_mtime_ns: int, source_size: int) -> tuple[int, ...]:
    # Metadata participates in the key so replacing a source file in place does
    # not reuse an old PTS index for the new content.
    del source_mtime_ns, source_size
    path = Path(path_text)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    values: list[int] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        if token and token != "N/A":
            try:
                values.append(int(token))
            except ValueError:
                continue
    return tuple(values)


def _frame_pts(path: Path) -> list[int]:
    try:
        source_stat = path.stat()
    except OSError:
        return []
    return list(_frame_pts_cached(str(path), int(source_stat.st_mtime_ns), int(source_stat.st_size)))


def frame_pts(path: str | Path) -> tuple[int, ...]:
    """Return the source video PTS sequence without decoding pixel frames."""

    return tuple(_frame_pts(Path(path)))


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    width: int
    height: int
    source_fps: float
    duration_s: float
    frame_count: int
    codec: str
    time_base: tuple[int, int]

    @classmethod
    def from_path(cls, path: str | Path) -> "VideoInfo":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        probed = _ffprobe(source, "width", "height", "r_frame_rate", "avg_frame_rate", "duration", "nb_frames", "codec_name", "time_base")
        stream = (probed or {}).get("streams", [{}])[0]
        format_data = (probed or {}).get("format", {})
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {source}")
        width = int(stream.get("width") or capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get("height") or capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _parse_ratio(stream.get("avg_frame_rate") or stream.get("r_frame_rate"), capture.get(cv2.CAP_PROP_FPS))
        duration = _parse_ratio(stream.get("duration") or format_data.get("duration"), 0.0)
        frame_count_value = stream.get("nb_frames")
        try:
            frame_count = int(frame_count_value) if frame_count_value not in (None, "N/A") else 0
        except ValueError:
            frame_count = 0
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        if frame_count <= 0:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = str(stream.get("codec_name") or "unknown")
        raw_time_base = str(stream.get("time_base") or "")
        if "/" in raw_time_base:
            num_text, den_text = raw_time_base.split("/", 1)
            try:
                time_base = (int(num_text), int(den_text))
            except ValueError:
                time_base = (1, max(1, round(fps)))
        else:
            time_base = (1, max(1, round(fps)))
        capture.release()
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(f"invalid video metadata for {source}")
        return cls(str(source), width, height, fps, duration, frame_count, codec, time_base)


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    frame_index: int
    pts: int
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.pts < 0:
            raise ValueError("pts must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def _scene_score(previous: np.ndarray, current: np.ndarray) -> float:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.resize(previous_gray, (64, 36), interpolation=cv2.INTER_AREA)
    current_gray = cv2.resize(current_gray, (64, 36), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)) / 255.0)


def scene_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Return the normalized grayscale difference used for cut scouting."""

    return _scene_score(previous, current)


def detect_shots(
    frames: Sequence[np.ndarray] | Iterable[np.ndarray],
    fps: float,
    manual_boundaries: Sequence[int] = (),
    scene_threshold: float = 0.28,
) -> list[ShotBoundary]:
    """Return ordered shot starts; manual boundaries always win at an index."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_list = frames if isinstance(frames, Sequence) else list(frames)
    if not frame_list:
        return []
    manual = {int(value) for value in manual_boundaries if 0 < int(value) < len(frame_list)}
    starts: dict[int, ShotBoundary] = {0: ShotBoundary(0, 0, "start", 1.0)}
    previous = frame_list[0]
    for index, frame in enumerate(frame_list[1:], start=1):
        if index in manual:
            starts[index] = ShotBoundary(index, index, "manual", 1.0)
        else:
            score = _scene_score(previous, frame)
            if score >= scene_threshold:
                starts[index] = ShotBoundary(index, index, "scene", min(1.0, score))
        previous = frame
    for index in sorted(manual):
        starts[index] = ShotBoundary(index, index, "manual", 1.0)
    return [starts[index] for index in sorted(starts)]


def shot_ranges(frame_count: int, boundaries: Sequence[ShotBoundary]) -> list[tuple[int, int]]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    starts = sorted({boundary.frame_index for boundary in boundaries if 0 <= boundary.frame_index < frame_count})
    if frame_count and 0 not in starts:
        starts.insert(0, 0)
    return [(start, end) for start, end in zip(starts, starts[1:] + [frame_count]) if start < end]


def iter_video_frames(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    *,
    pts_per_frame: float | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield frame index, source PTS, and BGR pixels without retaining the clip."""

    if start_frame < 0 or (end_frame is not None and end_frame < start_frame):
        raise ValueError("invalid frame range")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    # Exact ffprobe PTS extraction is useful for a full-source pass but can be
    # disproportionately expensive for a short window in a long recording.
    # Callers with a CFR source may provide the declared PTS/frame step; a full
    # pass still uses the source PTS index by default.
    pts_values = _frame_pts(source) if (start_frame == 0 and end_frame is None and pts_per_frame is None) else []
    if pts_per_frame is not None and (not np.isfinite(pts_per_frame) or pts_per_frame <= 0):
        raise ValueError("pts_per_frame must be positive and finite")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {source}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame
    else:
        index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start_frame and (end_frame is None or index < end_frame):
                pts = pts_values[index] if index < len(pts_values) else int(round(index * pts_per_frame)) if pts_per_frame is not None else index
                yield index, pts, frame
            index += 1
            if end_frame is not None and index >= end_frame:
                break
    finally:
        capture.release()
