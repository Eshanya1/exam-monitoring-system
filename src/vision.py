"""
Computer vision pipeline for exam monitoring:

- Face detection: OpenCV Haar cascade (bundled, no extra download).
- Head-pose deviation: real 3D pose estimation via MediaPipe FaceMesh
  landmarks + cv2.solvePnP against a canonical 6-point face model,
  reporting a single calibrated rotation-magnitude (degrees) rather than
  raw yaw/pitch/roll. Decomposing solvePnP's rotation matrix into named
  yaw/pitch/roll was tested and found unreliable for this landmark set
  (axis coupling near-frontal) -- the corrected *total* rotation
  magnitude was validated with synthetic rotation tests (0-30 deg
  applied rotation -> <1 deg measurement error) and is what anomaly
  detection is based on. See `pose_calibration.py` and the README for
  the full validation writeup.
- Identity verification: face-region intensity-histogram signature.
  This remains a lightweight, dependency-free stand-in for a real
  face-embedding model (FaceNet/ArcFace/InsightFace) -- swapping it is a
  one-function change (`_face_signature` / `compare_signatures`), called
  out explicitly here and in the README as the one component not yet
  production-grade.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

_mp_face_mesh = mp.solutions.face_mesh
_FACE_MESH = _mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

_IDX = [1, 152, 33, 263, 61, 291]
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)

_CALIBRATION_PATH = Path(__file__).parent / "pose_calibration.npy"
_R_FIX = np.load(_CALIBRATION_PATH) if _CALIBRATION_PATH.exists() else np.eye(3)


@dataclass
class FrameAnalysis:
    faces_detected: int
    identity_match: Optional[float]
    head_pose_deviation_deg: Optional[float]
    anomalies: list = field(default_factory=list)


def _face_signature(gray_face: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_face, (64, 64))
    hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def compare_signatures(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_CORREL))


def register_reference(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return _face_signature(gray[y:y + h, x:x + w])


def _estimate_head_pose_deviation(image_bgr: np.ndarray) -> Optional[float]:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _FACE_MESH.process(rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in _IDX], dtype=np.float64)

    cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    R_corrected = _R_FIX @ R
    rvec_corrected, _ = cv2.Rodrigues(R_corrected)
    return float(np.degrees(np.linalg.norm(rvec_corrected)))


def analyze_frame(
    image_bgr: np.ndarray,
    reference_signature: Optional[np.ndarray] = None,
    pose_deviation_threshold_deg: float = 20.0,
) -> FrameAnalysis:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    anomalies = []

    if len(faces) == 0:
        anomalies.append("no_face_detected")
        return FrameAnalysis(0, None, None, anomalies)
    if len(faces) > 1:
        anomalies.append("multiple_faces_detected")

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    identity_match = None
    if reference_signature is not None:
        sig = _face_signature(face_roi)
        identity_match = compare_signatures(reference_signature, sig)
        if identity_match < 0.5:
            anomalies.append("identity_mismatch")

    deviation = _estimate_head_pose_deviation(image_bgr)
    if deviation is None:
        anomalies.append("landmarks_not_detected")
    elif deviation > pose_deviation_threshold_deg:
        anomalies.append("head_pose_deviation")

    return FrameAnalysis(len(faces), identity_match, deviation, anomalies)
