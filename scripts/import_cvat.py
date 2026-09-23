#!/usr/bin/env python3
"""Import reviewed CVAT video tracks as an annotation manifest and MOT reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from football_tracking.annotations import AnnotationError, load_annotation_manifest, mot_reference_from_manifest
from football_tracking.cvat import CvatBridgeError, import_cvat_manifest
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="reviewed CVAT for video 1.1 XML")
    parser.add_argument("--provenance", type=Path, required=True, help="provenance.json from export_cvat.py")
    parser.add_argument("--source", type=Path, required=True, help="same source video used to create the CVAT task")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed", action="store_true", required=True, help="assert that all imported annotations have been reviewed")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True, help="ISO-8601 timestamp with timezone")
    args = parser.parse_args(argv)

    try:
        source_hash = sha256_file(args.source)
        info = VideoInfo.from_path(args.source)
        try:
            provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CvatBridgeError(f"unable to read provenance.json: {error}") from error
        if not isinstance(provenance, dict):
            raise CvatBridgeError("provenance.json must contain a JSON object")
        source = provenance.get("source", {})
        if not isinstance(source, dict):
            raise CvatBridgeError("provenance.json has invalid source metadata")
        if str(source.get("sha256", "")) != source_hash:
            raise CvatBridgeError("provenance source sha256 does not match --source")
        if (int(source.get("width", -1)), int(source.get("height", -1)), int(source.get("frame_count", -1))) != (info.width, info.height, info.frame_count):
            raise CvatBridgeError("provenance source dimensions/frame_count do not match --source")

        manifest_value = import_cvat_manifest(
            args.annotations.read_text(encoding="utf-8"),
            provenance,
            source_sha256=source_hash,
            pts_by_frame=frame_pts(args.source),
            reviewer=args.reviewer,
            reviewed_at=args.reviewed_at,
        )
        with TemporaryDirectory(prefix="football-cvat-import-") as temporary:
            manifest_path = Path(temporary) / "annotations.json"
            manifest_path.write_text(json.dumps(manifest_value), encoding="utf-8")
            parsed = load_annotation_manifest(manifest_path, source_hash, require_reviewed=True)
            reference = mot_reference_from_manifest(parsed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "annotations.json").write_text(json.dumps(manifest_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "mot-reference.json").write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, CvatBridgeError, AnnotationError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
