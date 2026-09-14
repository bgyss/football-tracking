"""Small deterministic JSONL cache for detector output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .detector import Detection


class CacheMismatch(ValueError):
    """Raised when cached detections do not match the requested source/config."""


class DetectionCache:
    @staticmethod
    def save(
        path: str | Path,
        source_hash: str,
        detector_config_hash: str,
        frames: Mapping[int, Sequence[Detection]],
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            metadata: dict[str, Any] = {"source_hash": source_hash, "detector_config_hash": detector_config_hash}
            if provenance is not None:
                metadata["provenance"] = dict(provenance)
            handle.write(json.dumps({"_meta": metadata}, sort_keys=True) + "\n")
            for frame_index in sorted(frames):
                row = {
                    "frame_index": int(frame_index),
                    "detections": [detection.to_dict() for detection in frames[frame_index]],
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    @staticmethod
    def load(path: str | Path, source_hash: str, detector_config_hash: str) -> dict[int, list[Detection]]:
        _, result = DetectionCache._read(path, source_hash, detector_config_hash)
        return result

    @staticmethod
    def load_strict(
        path: str | Path,
        source_hash: str,
        detector_config_hash: str,
        *,
        frame_count: int,
        provenance: Mapping[str, Any],
    ) -> dict[int, list[Detection]]:
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        metadata, result = DetectionCache._read(path, source_hash, detector_config_hash, reject_duplicates=True)
        actual_provenance = metadata.get("provenance")
        if not isinstance(actual_provenance, dict):
            raise CacheMismatch("cache provenance is missing")
        for key, expected in provenance.items():
            if actual_provenance.get(key) != expected:
                raise CacheMismatch(f"cache provenance {key} does not match")
        expected_indices = set(range(frame_count))
        observed_indices = set(result)
        invalid = sorted(observed_indices - expected_indices)
        missing = sorted(expected_indices - observed_indices)
        if invalid:
            raise CacheMismatch(f"cache contains out-of-range frame indices: {invalid[:5]}")
        if missing:
            raise CacheMismatch(f"cache is missing explicit frame records: {missing[:5]}")
        return result

    @staticmethod
    def merge_strict_parts(
        destination: str | Path,
        parts: Sequence[str | Path],
        source_hash: str,
        detector_config_hash: str,
        *,
        frame_count: int,
        provenance: Mapping[str, Any],
    ) -> dict[int, list[Detection]]:
        """Merge detector-only chunks, rejecting any provenance or frame overlap."""

        merged: dict[int, list[Detection]] = {}
        for part in parts:
            metadata, rows = DetectionCache._read(part, source_hash, detector_config_hash, reject_duplicates=True)
            if metadata.get("provenance") != dict(provenance):
                raise CacheMismatch(f"cache part provenance does not match: {part}")
            duplicate = set(merged).intersection(rows)
            if duplicate:
                raise CacheMismatch(f"cache parts contain duplicate frame records: {sorted(duplicate)[:5]}")
            merged.update(rows)
        expected = set(range(frame_count))
        if set(merged) != expected:
            missing = sorted(expected - set(merged))
            extra = sorted(set(merged) - expected)
            raise CacheMismatch(f"cache parts do not cover source frames; missing={missing[:5]} extra={extra[:5]}")
        DetectionCache.save(destination, source_hash, detector_config_hash, merged, provenance=provenance)
        return merged

    @staticmethod
    def _read(
        path: str | Path,
        source_hash: str,
        detector_config_hash: str,
        *,
        reject_duplicates: bool = False,
    ) -> tuple[dict[str, Any], dict[int, list[Detection]]]:
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
                frame_index = int(row["frame_index"])
                if reject_duplicates and frame_index in result:
                    raise CacheMismatch(f"cache contains duplicate frame record: {frame_index}")
                result[frame_index] = [Detection.from_dict(item) for item in row["detections"]]
        return dict(metadata), result
