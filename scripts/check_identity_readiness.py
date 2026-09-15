#!/usr/bin/env python3
"""Report whether reviewed calibration, timing, and identity inputs can support evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from football_tracking.annotations import AnnotationError, load_annotation_manifest
from football_tracking.calibration import calibration_quality_report, load_calibrations
from football_tracking.calibration_timeline import CalibrationTimelineError, load_calibration_timeline, timeline_quality_report
from football_tracking.evaluation import EvaluationError, load_cross_shot_identity, load_reviewed_mot_reference
from football_tracking.metrics import sha256_file
from football_tracking.replay import ReplayAlignmentError, load_play_alignments


def _entry(status: str, reason: str | None = None, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if reason:
        result["reason"] = reason
    result.update(values)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--play-alignment", type=Path)
    parser.add_argument("--reviewed-reference", type=Path)
    parser.add_argument("--review-pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"source video does not exist: {args.source}")
    source_hash = sha256_file(args.source)
    report: dict[str, Any] = {"schema_version": 1, "source": {"path": str(args.source), "sha256": source_hash}, "gates": {}}

    if args.review_pack is None:
        report["gates"]["review_pack"] = _entry("missing")
    else:
        try:
            value = json.loads(args.review_pack.read_text(encoding="utf-8"))
            declared = value.get("source", {}).get("sha256") if isinstance(value, dict) else None
            report["gates"]["review_pack"] = _entry("unreviewed" if value.get("reviewed") is False else "reviewed", source_hash_match=declared == source_hash)
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            report["gates"]["review_pack"] = _entry("invalid", str(error))

    if args.annotations is None:
        report["gates"]["annotations"] = _entry("missing")
    else:
        try:
            manifest = load_annotation_manifest(args.annotations, source_hash, require_reviewed=True)
            report["gates"]["annotations"] = _entry("valid", shots=len(manifest.shots), annotations=len(manifest.annotations), landmarks=len(manifest.landmarks), frame_labels=len(manifest.frame_labels))
        except (AnnotationError, OSError, ValueError) as error:
            report["gates"]["annotations"] = _entry("invalid", str(error))

    if args.calibration is None:
        report["gates"]["calibration"] = _entry("missing")
    else:
        try:
            value = json.loads(args.calibration.read_text(encoding="utf-8"))
            if isinstance(value, dict) and int(value.get("schema_version", 1)) >= 2:
                timeline = load_calibration_timeline(args.calibration, source_sha256=source_hash)
                quality = timeline_quality_report(timeline)
            else:
                quality = calibration_quality_report(load_calibrations(args.calibration, source_sha256=source_hash))
            report["gates"]["calibration"] = _entry(quality.get("status", "invalid"), quality=quality)
        except (CalibrationTimelineError, OSError, ValueError, json.JSONDecodeError) as error:
            report["gates"]["calibration"] = _entry("invalid", str(error))

    if args.play_alignment is None:
        report["gates"]["timing"] = _entry("missing")
    else:
        try:
            alignments = load_play_alignments(args.play_alignment, source_sha256=source_hash)
            eligible = [play.timing_eligible for play in alignments.plays]
            report["gates"]["timing"] = _entry("valid" if all(eligible) else "unvalidated", plays=len(eligible), eligible_plays=sum(eligible), alignments=[{"play_id": play.play_id, "timing_eligible": play.timing_eligible, "timing_report": play.timing_report} for play in alignments.plays])
        except (ReplayAlignmentError, OSError, ValueError, json.JSONDecodeError) as error:
            report["gates"]["timing"] = _entry("invalid", str(error))

    if args.reviewed_reference is None:
        report["gates"]["reference"] = _entry("missing")
    else:
        try:
            load_reviewed_mot_reference(args.reviewed_reference, source_sha256=source_hash)
            cross_map = load_cross_shot_identity(args.reviewed_reference, source_sha256=source_hash)
            report["gates"]["reference"] = _entry("valid" if cross_map else "missing_cross_shot_map", cross_shot_map=bool(cross_map))
        except (EvaluationError, OSError, ValueError, json.JSONDecodeError) as error:
            report["gates"]["reference"] = _entry("invalid", str(error))

    required = ("calibration", "timing", "reference")
    report["status"] = "ready_for_evaluation" if all(report["gates"][name]["status"] == "valid" for name in required) else "not_ready"
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
