#!/usr/bin/env python3
"""Propose motion-burst frames for manual pre-snap/snap/release event review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def motion_burst_proposals(
    source: Path,
    *,
    start_frame: int,
    end_frame: int,
    sample_step: int = 3,
    max_proposals: int = 8,
    minimum_separation_s: float = 0.5,
) -> dict:
    info = VideoInfo.from_path(source)
    if not 0 <= start_frame < end_frame <= info.frame_count:
        raise ValueError("frame range must satisfy 0 <= start < end <= source frame count")
    if sample_step < 1 or max_proposals < 1 or not math.isfinite(minimum_separation_s) or minimum_separation_s < 0:
        raise ValueError("invalid motion proposal policy")
    pts_values = frame_pts(source)
    if len(pts_values) != info.frame_count:
        raise ValueError("exact source PTS coverage is incomplete")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise ValueError("source PTS sequence must be strictly increasing")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open source video {source}")
    scores: list[tuple[int, float, float, float]] = []
    previous: np.ndarray | None = None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for source_frame in range(start_frame, end_frame):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"unable to decode source frame {source_frame}")
            if source_frame % sample_step != 0 and source_frame != start_frame:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
                global_motion = np.median(flow.reshape(-1, 2), axis=0)
                residual = np.linalg.norm(flow - global_motion[None, None, :], axis=2)
                residual_score = float(np.percentile(residual, 95))
                mean_residual = float(np.mean(residual))
                camera_motion_score = float(np.linalg.norm(global_motion))
                scores.append((source_frame, residual_score, mean_residual, camera_motion_score))
            previous = gray
    finally:
        capture.release()
    if not scores:
        return {"schema_version": 1, "reviewed": False, "source_sha256": sha256_file(source), "source": {"width": info.width, "height": info.height, "frame_count": info.frame_count, "time_base": list(info.time_base)}, "range": {"start_frame": start_frame, "end_frame": end_frame}, "proposal_policy": {"method": "farneback_residual_flow_burst_v1", "sample_step_frames": sample_step, "minimum_separation_s": minimum_separation_s, "note": "motion bursts are review candidates; they are not classified snap or ball-release events"}, "proposals": [], "camera_motion_keyframe_proposals": []}
    candidate_rows: list[tuple[float, int, float, float]] = []
    for index, (frame, p95, mean, _) in enumerate(scores):
        left = max(0, index - 4)
        right = min(len(scores), index + 5)
        neighborhood = [scores[offset][1] for offset in range(left, right) if offset != index]
        baseline = float(np.median(neighborhood)) if neighborhood else 0.0
        increase = max(0.0, p95 - baseline)
        normalized = increase / max(0.25, baseline)
        if index > 0 and index + 1 < len(scores) and p95 < scores[index - 1][1] and p95 < scores[index + 1][1]:
            continue
        candidate_rows.append((normalized, frame, p95, mean))
    candidate_rows.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, object]] = []
    minimum_gap_frames = int(round(minimum_separation_s * info.source_fps))
    for score, frame, p95, mean in candidate_rows:
        if any(abs(frame - int(item["source_frame"])) < minimum_gap_frames for item in selected):
            continue
        radius = int(round(0.5 * info.source_fps))
        window_start = max(start_frame, frame - radius)
        window_end = min(end_frame, frame + radius + 1)
        selected.append({
            "event_candidate_id": f"motion-{frame}",
            "candidate_kind": "motion_burst",
            "source_frame": frame,
            "source_pts": pts_values[frame],
            "window_start_frame": window_start,
            "window_end_frame": window_end,
            "window_start_pts": pts_values[window_start],
            "window_end_exclusive_pts": pts_values[window_end] if window_end < len(pts_values) else None,
            "residual_flow_p95_px_per_sample": round(p95, 4),
            "residual_flow_mean_px_per_sample": round(mean, 4),
            "burst_over_local_baseline": round(score, 4),
            "suggested_manual_labels": ["pre_snap", "snap", "ball_release"],
            "review_status": "unreviewed",
        })
        if len(selected) >= max_proposals:
            break
    camera_rows: list[tuple[float, int]] = []
    for index, (frame, _, _, camera_motion) in enumerate(scores):
        neighborhood = [scores[offset][3] for offset in range(max(0, index - 4), min(len(scores), index + 5)) if offset != index]
        baseline = float(np.median(neighborhood)) if neighborhood else 0.0
        if camera_motion > max(0.5, baseline * 1.5):
            camera_rows.append((camera_motion, frame))
    camera_rows.sort(key=lambda item: (-item[0], item[1]))
    camera_proposals: list[dict[str, object]] = []
    for motion, frame in camera_rows:
        if any(abs(frame - int(item["source_frame"])) < minimum_gap_frames for item in camera_proposals):
            continue
        camera_proposals.append({"source_frame": frame, "source_pts": pts_values[frame], "median_global_flow_px_per_sample": round(motion, 4), "proposal_kind": "camera_motion_keyframe_candidate", "review_status": "unreviewed"})
        if len(camera_proposals) >= max_proposals:
            break
    return {
        "schema_version": 1,
        "reviewed": False,
        "source_sha256": sha256_file(source),
        "source": {"width": info.width, "height": info.height, "frame_count": info.frame_count, "time_base": list(info.time_base)},
        "range": {"start_frame": start_frame, "end_frame": end_frame},
        "proposal_policy": {"method": "farneback_residual_flow_burst_v1", "sample_step_frames": sample_step, "minimum_separation_s": minimum_separation_s, "note": "motion bursts are review candidates; they are not classified snap or ball-release events"},
        "proposals": selected,
        "camera_motion_keyframe_proposals": camera_proposals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True, help="exclusive source frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=8)
    parser.add_argument("--minimum-separation-s", type=float, default=0.5)
    args = parser.parse_args()
    proposal = motion_burst_proposals(args.source, start_frame=args.start_frame, end_frame=args.end_frame, sample_step=args.sample_step, max_proposals=args.max_proposals, minimum_separation_s=args.minimum_separation_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
