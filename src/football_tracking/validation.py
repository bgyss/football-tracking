"""Consistency checks for exported tracking runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = (
    "analysis-config.json",
    "calibration.json",
    "calibration-quality.json",
    "tracklet-refinement.json",
    "identity-links.json",
    "tracking-evaluation.json",
    "review.json",
    "metrics.json",
    "run-manifest.json",
    "identities.json",
    "observations.csv",
    "observations.parquet",
    "trajectories.csv",
    "play-trajectories.csv",
    "shots.json",
)


def validate_run_artifacts(directory: str | Path) -> dict[str, Any]:
    """Validate cross-file invariants without evaluating tracking quality."""

    root = Path(directory)
    errors: list[str] = []
    values: dict[str, Any] = {}
    for filename in REQUIRED_ARTIFACTS:
        path = root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        if path.suffix == ".json":
            try:
                values[filename] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"invalid JSON {filename}: {error}")
    manifest = values.get("run-manifest.json", {})
    analysis_config = values.get("analysis-config.json", {})
    metrics = values.get("metrics.json", {})
    if manifest.get("config_hash") != analysis_config.get("analysis_hash"):
        errors.append("manifest config_hash does not match analysis-config analysis_hash")
    source_hash = manifest.get("input_sha256")
    if source_hash and analysis_config.get("source_sha256") != source_hash:
        errors.append("analysis-config source_sha256 does not match manifest input_sha256")
    if metrics.get("analysis", {}).get("analysis_hash") != analysis_config.get("analysis_hash"):
        errors.append("metrics analysis hash does not match analysis-config")
    if metrics.get("observation_count") is not None:
        try:
            with (root / "observations.csv").open(encoding="utf-8") as handle:
                observation_count = sum(1 for _ in handle) - 1
            if int(metrics["observation_count"]) != observation_count:
                errors.append("metrics observation_count does not match observations.csv")
        except (OSError, TypeError, ValueError):
            errors.append("unable to count observations.csv")
    identity_links = values.get("identity-links.json", {})
    identities = values.get("identities.json", {})
    if identity_links.get("tracklet_count") != len(identities):
        errors.append("identity-links tracklet_count does not match identities.json")
    if identity_links.get("player_id_count") != len(set(identities.values())):
        errors.append("identity-links player_id_count does not match identities.json")
    evaluation = values.get("tracking-evaluation.json", {})
    review = values.get("review.json", {})
    if evaluation.get("promotion_gate") != review.get("promotion_gate"):
        errors.append("review and tracking-evaluation promotion gates differ")
    calibration = values.get("calibration.json", {})
    if source_hash and calibration.get("source_sha256") not in {None, source_hash}:
        errors.append("calibration source_sha256 does not match manifest input_sha256")
    return {"schema_version": 1, "status": "valid" if not errors else "invalid", "directory": str(root), "checked_artifacts": [filename for filename in REQUIRED_ARTIFACTS if (root / filename).is_file()], "errors": errors}
