#!/usr/bin/env python3
"""Fit a validated PTS-scoped calibration timeline from a reviewed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.annotations import load_annotation_manifest
from football_tracking.calibration_timeline import timeline_from_landmark_records
from football_tracking.metrics import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_annotation_manifest(args.annotations, sha256_file(args.source), require_reviewed=True)
    if not manifest.landmarks:
        raise SystemExit("reviewed manifest contains no landmarks")
    timeline = timeline_from_landmark_records(manifest.landmarks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = timeline.to_dict()
    artifact["source_sha256"] = sha256_file(args.source)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
