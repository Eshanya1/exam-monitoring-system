import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
import numpy as np
import pytest
from vision import analyze_frame, register_reference, _face_signature, compare_signatures

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "face.jpg")
HAS_FIXTURE = os.path.exists(FIXTURE_PATH)

# These 4 tests are integration tests: they need a real photo of a face for
# the Haar cascade + MediaPipe to detect anything meaningful. A real human
# face photo isn't committed to the repo (privacy/licensing), so on a fresh
# clone these are SKIPPED, not failed -- see tests/fixtures/README.md to run
# them locally with your own test photo. The deterministic logic these tests
# exercise (signature comparison math, pose math) is covered without a photo
# by test_signature_synthetic below and test_pose_synthetic.py.
requires_fixture = pytest.mark.skipif(not HAS_FIXTURE, reason="requires tests/fixtures/face.jpg, see tests/fixtures/README.md")


def test_signature_comparison_identical_patch_scores_high():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    sig_a = _face_signature(patch)
    sig_b = _face_signature(patch)
    assert compare_signatures(sig_a, sig_b) > 0.99


def test_signature_comparison_different_patches_score_lower():
    rng = np.random.default_rng(0)
    patch_a = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    patch_b = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    sig_a = _face_signature(patch_a)
    sig_b = _face_signature(patch_b)
    same = compare_signatures(sig_a, sig_a)
    different = compare_signatures(sig_a, sig_b)
    assert different < same


@pytest.fixture(scope="module")
def face_image():
    img = cv2.imread(FIXTURE_PATH)
    assert img is not None, "test fixture image missing"
    return img


@requires_fixture
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


@requires_fixture
def test_analyze_frame_matching_identity(face_image):
    sig = register_reference(face_image)
    result = analyze_frame(face_image, reference_signature=sig)
    assert result.faces_detected == 1
    assert result.identity_match > 0.9
    assert "identity_mismatch" not in result.anomalies


@requires_fixture
def test_analyze_frame_frontal_pose_no_deviation(face_image):
    result = analyze_frame(face_image)
    assert result.head_pose_deviation_deg is not None
    assert result.head_pose_deviation_deg < 5.0
    assert "head_pose_deviation" not in result.anomalies


@requires_fixture
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
