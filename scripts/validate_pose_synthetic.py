"""Synthetic rotation-ground-truth validation for the pose-estimation pipeline.

Bypasses MediaPipe and any real photo entirely: constructs synthetic 2D image
points by projecting the same 3D face model the production code uses
(vision._MODEL_POINTS) at a precisely known rotation, then feeds those points
through the *exact* production function (vision.pose_deviation_from_image_points,
including the real calibration matrix from pose_calibration.npy) and measures
how far the recovered rotation is from the angle that was actually applied.

IMPORTANT HONESTY NOTE: a noise-free version of this test measures ~0.00 deg
error at every angle -- that's expected (solvePnP recovering an exact
rotation from an exact, noiseless projection is a solved problem) but it only
proves the math/calibration composition is correct, not anything about
real-world accuracy. MediaPipe's actual landmark localization has real
pixel-level jitter, and that's the error source that matters in practice. So
this script injects Gaussian pixel noise (LANDMARK_NOISE_STD_PX, chosen as a
representative few-pixel jitter magnitude, not measured from real footage --
that would require real photos, which this script deliberately avoids) into
the projected points and reports the resulting error distribution over many
trials. That's what the README table actually reports: sensitivity to
plausible detector jitter, not an empirical measurement of MediaPipe itself.

Run it yourself to reproduce:

    python scripts/validate_pose_synthetic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from vision import _MODEL_POINTS, _R_FIX, pose_deviation_from_image_points

IMG_W, IMG_H = 640, 480
CAM_MATRIX = np.array([[IMG_W, 0, IMG_W / 2], [0, IMG_W, IMG_H / 2], [0, 0, 1]], dtype=np.float64)
TVEC = np.array([[0.0], [0.0], [1000.0]])  # object placed ~1m in front of the synthetic camera
LANDMARK_NOISE_STD_PX = 1.5  # representative per-landmark pixel jitter, see honesty note above
TRIALS_PER_ANGLE = 200
RNG = np.random.default_rng(seed=42)

# R_FIX maps "the pose the calibration photo was taken at" -> identity. So the
# pose that itself measures as 0 deviation is R_FIX's inverse (its transpose,
# since it's a rotation matrix) -- this is the pipeline's own "frontal" pose,
# not an arbitrary choice.
R_FRONTAL = _R_FIX.T

AXES = {
    "yaw (Y-axis)": np.array([0.0, 1.0, 0.0]),
    "pitch (X-axis)": np.array([1.0, 0.0, 0.0]),
    "roll (Z-axis)": np.array([0.0, 0.0, 1.0]),
}
TEST_ANGLES_DEG = [0, 5, 10, 20, 30]


def rotate(axis: np.ndarray, degrees: float) -> np.ndarray:
    rvec = axis * np.radians(degrees)
    R, _ = cv2.Rodrigues(rvec)
    return R


def measure(applied_deg: float, axis: np.ndarray, noisy: bool) -> float:
    R_test = rotate(axis, applied_deg) @ R_FRONTAL
    rvec_test, _ = cv2.Rodrigues(R_test)
    image_points, _ = cv2.projectPoints(_MODEL_POINTS, rvec_test, TVEC, CAM_MATRIX, np.zeros((4, 1)))
    image_points = image_points.reshape(-1, 2)
    if noisy:
        image_points = image_points + RNG.normal(0, LANDMARK_NOISE_STD_PX, image_points.shape)
    measured = pose_deviation_from_image_points(image_points, CAM_MATRIX)
    return measured


def main():
    print("-- Noise-free sanity check (validates the math, not real-world accuracy) --")
    for axis_name, axis in AXES.items():
        for angle in TEST_ANGLES_DEG:
            measured = measure(angle, axis, noisy=False)
            print(f"{axis_name:<16}{angle:>9}°  ->  {measured:>7.4f}° (error {abs(measured-angle):.4f}°)")

    print()
    print(f"-- Under injected landmark noise (sigma={LANDMARK_NOISE_STD_PX}px, "
          f"{TRIALS_PER_ANGLE} trials/angle) --")
    print(f"{'Axis':<16}{'Applied':>10}{'Mean measured':>16}{'Mean error':>13}{'Max error':>12}")
    all_errors = []
    for axis_name, axis in AXES.items():
        for angle in TEST_ANGLES_DEG:
            measured_vals = [measure(angle, axis, noisy=True) for _ in range(TRIALS_PER_ANGLE)]
            errors = [abs(m - angle) for m in measured_vals]
            all_errors.extend(errors)
            mean_measured = sum(measured_vals) / len(measured_vals)
            print(f"{axis_name:<16}{angle:>9}°{mean_measured:>15.2f}°{sum(errors)/len(errors):>12.2f}°"
                  f"{max(errors):>11.2f}°")

    print()
    print(f"Overall mean error: {sum(all_errors)/len(all_errors):.2f}°")
    print(f"Overall max error: {max(all_errors):.2f}°")


if __name__ == "__main__":
    main()
