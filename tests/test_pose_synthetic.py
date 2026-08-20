"""Unit tests for the pose-estimation math using synthetic (known-rotation)
landmark points -- no MediaPipe, no real photo. See
scripts/validate_pose_synthetic.py for the full sensitivity sweep this is
a lighter-weight, CI-friendly version of; run that script directly to
reproduce the numbers quoted in README.md.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import cv2
import numpy as np
import pytest

from vision import _MODEL_POINTS, pose_deviation_from_image_points
from validate_pose_synthetic import CAM_MATRIX, TVEC, R_FRONTAL, rotate


def _synthetic_measurement(applied_deg, axis, noise_std_px=0.0, seed=0):
    R_test = rotate(axis, applied_deg) @ R_FRONTAL
    rvec_test, _ = cv2.Rodrigues(R_test)
    image_points, _ = cv2.projectPoints(_MODEL_POINTS, rvec_test, TVEC, CAM_MATRIX, np.zeros((4, 1)))
    image_points = image_points.reshape(-1, 2)
    if noise_std_px:
        rng = np.random.default_rng(seed)
        image_points = image_points + rng.normal(0, noise_std_px, image_points.shape)
    return pose_deviation_from_image_points(image_points, CAM_MATRIX)


def test_zero_rotation_measures_near_zero():
    measured = _synthetic_measurement(0.0, np.array([0.0, 1.0, 0.0]))
    assert measured < 0.01


@pytest.mark.parametrize("angle", [5, 10, 20, 30])
def test_noise_free_recovery_matches_applied_angle(angle):
    measured = _synthetic_measurement(angle, np.array([0.0, 1.0, 0.0]))
    assert abs(measured - angle) < 0.01


@pytest.mark.parametrize("angle", [0, 5, 10, 20, 30])
def test_under_realistic_landmark_noise_error_stays_bounded(angle):
    # sigma=1.5px matches scripts/validate_pose_synthetic.py's stated assumption.
    # Bound (3.5 deg) is set from the actual measured worst case (2.6 deg) plus
    # margin, not backed into an assumed number -- see that script's full sweep.
    errors = [
        abs(_synthetic_measurement(angle, np.array([0.0, 1.0, 0.0]), noise_std_px=1.5, seed=i) - angle)
        for i in range(50)
    ]
    assert max(errors) < 3.5
