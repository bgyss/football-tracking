#!/usr/bin/env python3
"""Create unreviewed Tesseract jersey-number cues from high-resolution track crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import cv2

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def _read_words(tsv: str, min_confidence: float) -> list[dict[str, object]]:
    lines = tsv.splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    text_index, conf_index = headers.index("text"), headers.index("conf")
    candidates: dict[int, float] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) <= max(text_index, conf_index):
            continue
        digits = re.findall(r"(?<!\d)\d{1,2}(?!\d)", fields[text_index])
        try:
            confidence = float(fields[conf_index]) / 100.0
        except ValueError:
            continue
        if not 0.0 <= confidence <= 1.0 or confidence < min_confidence:
            continue
        for token in digits:
            number = int(token)
            candidates[number] = max(candidates.get(number, 0.0), confidence)
    return [{"number": number, "confidence": round(confidence, 4)} for number, confidence in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))]


def _ocr(crop_bgr, executable: str, min_confidence: float) -> tuple[list[dict[str, object]], str]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", enlarged)
    if not ok:
        return [], "image_encoding_failed"
    command = [executable, "stdin", "stdout", "--psm", "11", "-c", "tessedit_char_whitelist=0123456789", "tsv"]
    completed = subprocess.run(command, input=encoded.tobytes(), capture_output=True, check=False)
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "tesseract failed")
    candidates = _read_words(completed.stdout.decode("utf-8", errors="replace"), min_confidence)
    return candidates, "read" if candidates else "no_confident_digits"


def propose_jersey_reads(
    source_path: Path,
    observations_path: Path,
    analysis_config_path: Path,
    output_path: Path,
    *,
    tesseract: str = "tesseract",
    min_confidence: float = 0.35,
    min_box_height_px: float = 55.0,
    sample_stride_frames: int = 12,
    max_samples_per_tracklet: int = 24,
) -> None:
    executable = shutil.which(tesseract)
    if executable is None:
        raise FileNotFoundError(f"Tesseract executable not found: {tesseract}")
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0 or not math.isfinite(min_box_height_px) or min_box_height_px <= 0 or sample_stride_frames < 1 or max_samples_per_tracklet < 1:
        raise ValueError("invalid OCR sampling policy")
    info = VideoInfo.from_path(source_path)
    source_hash = sha256_file(source_path)
    pts_values = frame_pts(source_path)
    if len(pts_values) != info.frame_count:
        raise ValueError("exact source PTS coverage is incomplete")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise ValueError("source PTS sequence must be strictly increasing")
    analysis_config = json.loads(analysis_config_path.read_text(encoding="utf-8"))
    if not isinstance(analysis_config, dict) or analysis_config.get("source_sha256") != source_hash:
        raise ValueError("analysis config source hash does not match source video")
    analysis_hash = str(analysis_config.get("base_analysis_hash") or analysis_config.get("analysis_hash") or "")
    if len(analysis_hash) != 16 or any(character not in "0123456789abcdef" for character in analysis_hash.lower()):
        raise ValueError("analysis config must declare a 16-character base_analysis_hash")
    observations_hash = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    version_result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    tesseract_version = version_result.stdout.strip().splitlines()[0] if version_result.returncode == 0 and version_result.stdout.strip() else "unknown"
    by_tracklet: dict[str, list[dict[str, object]]] = {}
    with observations_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tracklet_id = str(row.get("tracklet_id", "")).strip()
            if not tracklet_id:
                continue
            frame = int(row["frame_index"])
            pts = int(row["pts"])
            if frame < 0 or frame >= len(pts_values) or pts != pts_values[frame]:
                raise ValueError(f"observation PTS mismatch for {tracklet_id}@{frame}")
            if json.loads(row["time_base"]) != list(info.time_base):
                raise ValueError(f"observation time_base mismatch for {tracklet_id}@{frame}")
            bbox = tuple(float(value) for value in json.loads(row["bbox_xyxy_px"]))
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"invalid box for {tracklet_id}@{frame}")
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > info.width or bbox[3] > info.height:
                raise ValueError(f"observation box lies outside source frame for {tracklet_id}@{frame}")
            score = float(row.get("detection_score") or 0.0)
            by_tracklet.setdefault(tracklet_id, []).append({"tracklet_id": tracklet_id, "source_frame": frame, "source_pts": pts, "bbox_xyxy_px": bbox, "detection_score": score})

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {source_path}")
    tracklets: dict[str, dict[str, object]] = {}
    try:
        for tracklet_id, rows in sorted(by_tracklet.items()):
            bins: dict[int, dict[str, object]] = {}
            for row in rows:
                bbox = tuple(float(value) for value in row["bbox_xyxy_px"])
                if bbox[3] - bbox[1] < min_box_height_px:
                    continue
                bucket = int(row["source_frame"]) // sample_stride_frames
                previous = bins.get(bucket)
                quality = (float(row["detection_score"]), bbox[3] - bbox[1], -int(row["source_frame"]))
                if previous is None:
                    bins[bucket] = row
                else:
                    previous_box = tuple(float(value) for value in previous["bbox_xyxy_px"])
                    previous_quality = (float(previous["detection_score"]), previous_box[3] - previous_box[1], -int(previous["source_frame"]))
                    if quality > previous_quality:
                        bins[bucket] = row
            candidates = [bins[index] for index in sorted(bins)]
            if len(candidates) > max_samples_per_tracklet:
                positions = [round(index * (len(candidates) - 1) / (max_samples_per_tracklet - 1)) for index in range(max_samples_per_tracklet)] if max_samples_per_tracklet > 1 else [len(candidates) // 2]
                candidates = [candidates[index] for index in sorted(set(positions))]
            selected: list[dict[str, object]] = []
            for row in candidates:
                frame_index = int(row["source_frame"])
                bbox = tuple(float(value) for value in row["bbox_xyxy_px"])
                height = bbox[3] - bbox[1]
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"unable to decode source frame {frame_index}")
                reported_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if reported_frame != frame_index:
                    raise RuntimeError(f"video seek returned frame {reported_frame}, expected {frame_index}")
                if pts_values[frame_index] != int(row["source_pts"]):
                    raise RuntimeError(f"source PTS mismatch at frame {frame_index}")
                x1, y1, x2, y2 = bbox
                crop_x1 = max(0, int(round(x1 + (x2 - x1) * 0.08)))
                crop_x2 = min(frame.shape[1], int(round(x2 - (x2 - x1) * 0.08)))
                crop_y1 = max(0, int(round(y1 + (y2 - y1) * 0.12)))
                crop_y2 = min(frame.shape[0], int(round(y1 + (y2 - y1) * 0.78)))
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                candidates, status = _ocr(crop, executable, min_confidence)
                selected.append({
                    "source_frame": frame_index,
                    "source_pts": int(row["source_pts"]),
                    "bbox_xyxy_px": list(bbox),
                    "crop_xyxy_px": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "crop_quality": {"crop_width_px": crop_x2 - crop_x1, "crop_height_px": crop_y2 - crop_y1, "bbox_height_px": round(height, 3), "detection_score": float(row["detection_score"]), "laplacian_variance": round(sharpness, 3)},
                    "review_status": "unreviewed",
                    "model": tesseract_version,
                    "status": status,
                    "candidates": candidates,
                })
            if selected:
                tracklets[tracklet_id] = {"jersey_reads": selected, "appearance_embeddings": []}
    finally:
        capture.release()
    artifact = {
        "schema_version": 1,
        "reviewed": False,
        "source_sha256": source_hash,
        "analysis_hash": analysis_hash,
        "observations_sha256": observations_hash,
        "tracklets": tracklets,
        "proposal_policy": {"model": "tesseract", "version": tesseract_version, "min_confidence": min_confidence, "min_box_height_px": min_box_height_px, "sample_stride_frames": sample_stride_frames, "max_samples_per_tracklet": max_samples_per_tracklet},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="raw OCR confidence in [0,1]")
    parser.add_argument("--min-box-height-px", type=float, default=55.0)
    parser.add_argument("--sample-stride-frames", type=int, default=12)
    parser.add_argument("--max-samples-per-tracklet", type=int, default=24)
    args = parser.parse_args()
    propose_jersey_reads(args.source, args.observations, args.analysis_config, args.output, tesseract=args.tesseract, min_confidence=args.min_confidence, min_box_height_px=args.min_box_height_px, sample_stride_frames=args.sample_stride_frames, max_samples_per_tracklet=args.max_samples_per_tracklet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
