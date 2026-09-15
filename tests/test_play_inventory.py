from __future__ import annotations

import json

import cv2
import numpy as np

from scripts.build_play_inventory import main


def test_build_play_inventory_marks_candidates_unreviewed(tmp_path, monkeypatch) -> None:
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((24, 32, 3), 20 if index < 2 else 220, dtype=np.uint8))
    writer.release()
    output = tmp_path / "inventory.json"
    monkeypatch.setattr("sys.argv", ["build_play_inventory.py", "--input", str(video), "--output", str(output), "--manual-cut", "2"])
    assert main() == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["reviewed"] is False
    assert len(value["shots"]) == 2
    assert value["shots"][0]["review_status"] == "unreviewed"


def test_build_play_inventory_detects_a_local_cut_spike(tmp_path, monkeypatch) -> None:
    video = tmp_path / "auto-cut.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((24, 32, 3), 20 if index < 3 else 230, dtype=np.uint8))
    writer.release()
    output = tmp_path / "auto-inventory.json"
    monkeypatch.setattr("sys.argv", ["build_play_inventory.py", "--input", str(video), "--output", str(output)])
    assert main() == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert len(value["shots"]) >= 2
    assert value["scouting_policy"]["scene_threshold"] == 0.08
