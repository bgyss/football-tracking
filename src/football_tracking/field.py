"""Canonical NFL field coordinates and semantic landmark parsing."""

from __future__ import annotations

import re
import math

FIELD_LENGTH_YARDS = 120.0
FIELD_WIDTH_YARDS = 160.0 / 3.0
GOAL_LINE_X = (10.0, 110.0)
HASH_NEAR_YARDS = 70.75 / 3.0
HASH_FAR_YARDS = FIELD_WIDTH_YARDS - HASH_NEAR_YARDS


def field_landmark(landmark_id: str) -> tuple[float, float]:
    """Return a canonical point for a semantic field landmark.

    Coordinates use x=0 at the west end line and y=0 at the near sideline.
    The parser intentionally requires a side/row for repeated markings so a
    visually plausible but mirrored calibration cannot silently pass.

    Supported forms include ``yardline:20:sideline:near``,
    ``yardline:20:hash:far``, ``goal_line:west:hash:near`` and
    ``end_line:east:sideline:far``.
    """

    if not isinstance(landmark_id, str) or not landmark_id.strip():
        raise ValueError("landmark id must be non-empty")
    tokens = [token.strip().lower() for token in re.split(r"[:/\-]+", landmark_id) if token.strip()]
    if len(tokens) < 3:
        raise ValueError(f"landmark id is not semantic enough: {landmark_id!r}")
    kind = tokens[0].replace("_", "")
    if tokens[0] in {"goal", "end"} and len(tokens) > 1 and tokens[1].replace("_", "") in {"line", "lines"}:
        tokens = [tokens[0]] + tokens[2:]
    if kind in {"yardline", "yard"}:
        try:
            x = float(tokens[1])
        except ValueError as error:
            raise ValueError(f"invalid yardline in {landmark_id!r}") from error
        if not math.isfinite(x) or not 0.0 <= x <= FIELD_LENGTH_YARDS:
            raise ValueError("yardline must lie within the 120-yard field")
        row = tokens[-1]
        if len(tokens) >= 4 and tokens[-2] in {"hash", "hashmark", "hashmarks"}:
            y = HASH_NEAR_YARDS if row in {"near", "home", "a"} else HASH_FAR_YARDS if row in {"far", "away", "b"} else None
        else:
            y = {"near": 0.0, "sideline": 0.0, "far": FIELD_WIDTH_YARDS, "hash_near": HASH_NEAR_YARDS, "hash_far": HASH_FAR_YARDS}.get(row)
        if y is None:
            raise ValueError(f"landmark row must identify near/far sideline or hash: {landmark_id!r}")
        return (x, float(y))
    if kind in {"goal", "goalline", "goal_line", "endline", "end_line"}:
        if kind.startswith("goal"):
            if len(tokens) < 4:
                raise ValueError(f"goal-line landmark needs end and row: {landmark_id!r}")
            end = tokens[1]
            x = 10.0 if end in {"west", "left", "home", "a"} else 110.0 if end in {"east", "right", "away", "b"} else None
        else:
            if len(tokens) < 4:
                raise ValueError(f"end-line landmark needs end and row: {landmark_id!r}")
            end = tokens[1]
            x = 0.0 if end in {"west", "left", "home", "a"} else 120.0 if end in {"east", "right", "away", "b"} else None
        if x is None:
            raise ValueError(f"unknown field end in {landmark_id!r}")
        row = tokens[-1]
        if len(tokens) >= 4 and tokens[-2] in {"hash", "hashmark", "hashmarks"}:
            if row in {"near", "home", "a"}:
                return (x, HASH_NEAR_YARDS)
            if row in {"far", "away", "b"}:
                return (x, HASH_FAR_YARDS)
        if row in {"near", "sideline"}:
            return (x, 0.0)
        if row == "far":
            return (x, FIELD_WIDTH_YARDS)
        if row in {"hash_near", "hash", "home"}:
            return (x, HASH_NEAR_YARDS)
        if row in {"hash_far", "away"}:
            return (x, HASH_FAR_YARDS)
    raise ValueError(f"unsupported landmark id: {landmark_id!r}")


def validate_field_point(point: tuple[float, float] | list[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("field point must contain x and y")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= FIELD_LENGTH_YARDS or not 0.0 <= y <= FIELD_WIDTH_YARDS:
        raise ValueError("field point lies outside the 120 by 53 1/3 yard field")
    return (x, y)
