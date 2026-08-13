"""
One-time calibration utility: derives the fixed rotation-correction matrix
needed because the canonical 3D face model used for solvePnP (standard
6-point model from the classic OpenCV head-pose tutorial) is not aligned
with "camera looking at a frontal face = identity rotation". This is a
known property of PnP with a small, symmetric landmark set -- decomposing
the raw rotation matrix into yaw/pitch/roll directly gives a large,
consistent (non-random) offset even for genuinely frontal faces, and
individual-axis decomposition remains coupled/unreliable even after
correction (validated empirically -- see README). The corrected *total*
rotation magnitude, however, tracks applied rotation angle almost 1:1 in
testing (0-30 deg synthetic rotation -> <1 deg error), so that is what
`vision.py` uses for anomaly detection rather than named yaw/pitch/roll.

Run this once against a clear frontal reference photo to (re)generate
pose_calibration.npy. A default calibration (derived from a neutral
frontal test image) ships with the repo, so this step is optional unless
you want to recalibrate against your own camera/model-point setup.
"""
import sys
import cv2
import numpy as np
import mediapipe as mp

IDX = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


def calibrate(frontal_image_path: str, out_path: str = "pose_calibration.npy"):
    img = cv2.imread(frontal_image_path)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    result = fm.process(rgb)
    if not result.multi_face_landmarks:
        raise ValueError("No face detected in calibration image")

    lm = result.multi_face_landmarks[0].landmark
    image_points = np.array([(lm[i].x * w, lm[i].y * h) for i in IDX], dtype=np.float64)
    cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnP failed on calibration image")

    R_frontal, _ = cv2.Rodrigues(rvec)
    R_fix = R_frontal.T  # inverse rotation: maps this frontal pose back to identity
    np.save(out_path, R_fix)
    print(f"Saved calibration matrix -> {out_path}")
    return R_fix


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_face.jpg"
    calibrate(path)
