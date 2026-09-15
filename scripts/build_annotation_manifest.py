#!/usr/bin/env python3
"""Create an unreviewed annotation-manifest template from a review pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.annotations import manifest_template_from_review_pack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot", action="append", default=[], help="shot_id:start_frame:end_frame:camera_label[:play_id[:split]]; repeat")
    args = parser.parse_args()
    pack = json.loads(args.review_pack.read_text(encoding="utf-8"))
    shots: dict[str, dict[str, object]] = {}
    for value in args.shot:
        parts = value.split(":")
        if len(parts) < 4:
            raise SystemExit("--shot needs shot_id:start_frame:end_frame:camera_label[:play_id[:split]]")
        shot_id, start, end, camera = parts[:4]
        shots[shot_id] = {"start_frame": int(start), "end_frame": int(end), "camera_label": camera, "play_id": parts[4] if len(parts) > 4 and parts[4] else None, "split": parts[5] if len(parts) > 5 and parts[5] else "unassigned"}
    template = manifest_template_from_review_pack(pack, shots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
