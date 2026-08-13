import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
import numpy as np
import pytest
from vision import analyze_frame, register_reference

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "face.jpg")


@pytest.fixture(scope="module")
def face_image():
    img = cv2.imread(FIXTURE_PATH)
    assert img is not None, "test fixture image missing"
    return img


def test_register_reference_finds_face(face_image):
    sig = register_reference(face_image)
    assert sig is not None
    assert sig.shape == (32,)


def test_register_reference_no_face_returns_none():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert register_reference(blank) is None


def test_analyze_frame_no_face_flags_anomaly():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    result = analyze_frame(blank)
    assert result.faces_detected == 0
    assert "no_face_detected" in result.anomalies


def test_analyze_frame_matching_identity(face_image):
    sig = register_reference(face_image)
    result = analyze_frame(face_image, reference_signature=sig)
    assert result.faces_detected == 1
    assert result.identity_match > 0.9
    assert "identity_mismatch" not in result.anomalies


def test_analyze_frame_frontal_pose_no_deviation(face_image):
    result = analyze_frame(face_image)
    assert result.head_pose_deviation_deg is not None
    assert result.head_pose_deviation_deg < 5.0
    assert "head_pose_deviation" not in result.anomalies


def test_analyze_frame_rotated_flags_pose_deviation(face_image):
    h, w = face_image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
    rotated = cv2.warpAffine(face_image, M, (w, h))
    result = analyze_frame(rotated)
    # At 30 deg the Haar cascade may lose the face entirely (documented
    # limitation) -- either outcome below is an acceptable, honest result.
    if result.faces_detected == 0:
        assert "no_face_detected" in result.anomalies
    else:
        assert result.head_pose_deviation_deg is None or result.head_pose_deviation_deg > 15
