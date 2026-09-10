"""Small deterministic JSONL cache for detector output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .detector import Detection


class CacheMismatch(ValueError):
    """Raised when cached detections do not match the requested source/config."""


class DetectionCache:
    @staticmethod
    def save(path: str | Path, source_hash: str, detector_config_hash: str, frames: Mapping[int, Sequence[Detection]]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"_meta": {"source_hash": source_hash, "detector_config_hash": detector_config_hash}}, sort_keys=True) + "\n")
            for frame_index in sorted(frames):
                row = {
                    "frame_index": int(frame_index),
                    "detections": [detection.to_dict() for detection in frames[frame_index]],
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    @staticmethod
    def load(path: str | Path, source_hash: str, detector_config_hash: str) -> dict[int, list[Detection]]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open(encoding="utf-8") as handle:
            try:
                metadata = json.loads(next(handle))["_meta"]
            except (StopIteration, KeyError, json.JSONDecodeError) as error:
                raise CacheMismatch("cache metadata is missing") from error
            if metadata.get("source_hash") != source_hash:
                raise CacheMismatch("source hash does not match cache")
            if metadata.get("detector_config_hash") != detector_config_hash:
                raise CacheMismatch("detector configuration does not match cache")
            result: dict[int, list[Detection]] = {}
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                result[int(row["frame_index"])] = [Detection.from_dict(item) for item in row["detections"]]
        return result
