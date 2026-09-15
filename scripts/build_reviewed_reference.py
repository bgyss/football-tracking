#!/usr/bin/env python3
"""Convert a reviewed annotation manifest into the MOT/reference contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.annotations import load_annotation_manifest, mot_reference_from_manifest
from football_tracking.metrics import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_annotation_manifest(args.annotations, sha256_file(args.source), require_reviewed=True)
    reference = mot_reference_from_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
