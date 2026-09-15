from __future__ import annotations

import cv2
import numpy as np

from scripts.build_identity_review_pack import field_line_proposals


def test_field_line_proposals_are_unreviewed_hints() -> None:
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    cv2.line(image, (10, 20), (130, 20), (255, 255, 255), 3)
    proposals = field_line_proposals(image)
    assert proposals
    assert all(item["review_status"] == "unreviewed" for item in proposals)
