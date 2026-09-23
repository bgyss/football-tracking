#!/usr/bin/env python3
"""Convert a reviewed annotation manifest into the MOT/reference contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.annotations import load_annotation_manifest, mot_reference_from_manifest
from football_tracking.calibration_timeline import load_calibration_timeline
from football_tracking.metrics import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="optional source-hashed schema-v2 timeline to project reviewed source-pixel contacts into field yards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = sha256_file(args.source)
    manifest = load_annotation_manifest(args.annotations, source_hash, require_reviewed=True)
    calibration = load_calibration_timeline(args.calibration, source_sha256=source_hash) if args.calibration else None
    reference = mot_reference_from_manifest(manifest, calibration_timeline=calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
