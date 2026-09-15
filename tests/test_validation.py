from __future__ import annotations

import json

from football_tracking.validation import validate_run_artifacts


def test_validate_run_artifacts_reports_cross_file_hash_and_count_errors(tmp_path) -> None:
    (tmp_path / "analysis-config.json").write_text(json.dumps({"analysis_hash": "a", "source_sha256": "source"}), encoding="utf-8")
    (tmp_path / "run-manifest.json").write_text(json.dumps({"config_hash": "b", "input_sha256": "source"}), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"analysis": {"analysis_hash": "a"}}), encoding="utf-8")
    (tmp_path / "identity-links.json").write_text(json.dumps({"tracklet_count": 1, "player_id_count": 1}), encoding="utf-8")
    (tmp_path / "identities.json").write_text(json.dumps({"t1": "P01", "t2": "P01"}), encoding="utf-8")
    (tmp_path / "tracking-evaluation.json").write_text(json.dumps({"promotion_gate": {"status": "not_evaluated"}}), encoding="utf-8")
    (tmp_path / "review.json").write_text(json.dumps({"promotion_gate": {"status": "failed"}}), encoding="utf-8")
    report = validate_run_artifacts(tmp_path)
    assert report["status"] == "invalid"
    assert "manifest config_hash does not match analysis-config analysis_hash" in report["errors"]
    assert "identity-links tracklet_count does not match identities.json" in report["errors"]
