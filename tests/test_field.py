from __future__ import annotations

import pytest

from football_tracking.field import FIELD_WIDTH_YARDS, HASH_FAR_YARDS, HASH_NEAR_YARDS, field_landmark


def test_nfl_landmarks_have_fixed_orientation() -> None:
    assert field_landmark("yardline:20:sideline:near") == (20.0, 0.0)
    assert field_landmark("yardline:20:hash:near") == (20.0, HASH_NEAR_YARDS)
    assert field_landmark("yardline:20:hash:far") == (20.0, HASH_FAR_YARDS)
    assert field_landmark("goal_line:west:sideline:near") == (10.0, 0.0)
    assert field_landmark("end_line:east:sideline:far") == (120.0, FIELD_WIDTH_YARDS)


def test_field_landmark_rejects_ambiguous_or_invalid_markings() -> None:
    with pytest.raises(ValueError):
        field_landmark("yardline:20")
    with pytest.raises(ValueError):
        field_landmark("yardline:130:hash:near")
