# Real-Time AI Examination Monitoring System

A computer vision platform for identity verification, real 3D head-pose
tracking, and behavioral anomaly detection during online examinations,
exposed as an authenticated Flask REST API with persistent session state.

## Architecture

```
Proctor login (JWT) ──▶ /sessions/{id}/register ──▶ reference face signature (SQLite)
                                                              │
Webcam frame ──▶ /sessions/{id}/analyze ──▶ vision.py         │
                    │                        ├─ Haar cascade face detection
                    │                        ├─ MediaPipe FaceMesh landmarks
                    │                        │    → solvePnP → calibrated
                    │                        │      head-pose deviation (deg)
                    │                        └─ histogram identity match
                    ▼
              anomaly flags ──▶ SQLite (persisted) ──▶ /sessions/{id}/summary
```

## What's real here (and what's a documented stand-in)

**Genuinely solved, not a toy:**
- **Head-pose estimation** uses real 3D geometry (MediaPipe FaceMesh landmarks
  + `cv2.solvePnP` against a canonical face model) — not a heuristic. While
  building this I found that raw yaw/pitch/roll decomposition from the
  rotation matrix is unreliable for this landmark set (axis coupling for
  near-frontal poses — confirmed across 3 different PnP solvers). Fixed with
  a calibration step (`pose_calibration.py`) and validated the **corrected
  rotation magnitude** against synthetic ground truth — reproducible with
  `python scripts/validate_pose_synthetic.py` (also covered by
  `tests/test_pose_synthetic.py`), which projects the same 3D model the
  production code uses at a precisely known rotation and measures recovery
  error, both noise-free (validates the math) and under injected landmark
  jitter (validates real-world-relevant sensitivity):

  | Applied rotation | Noise-free error | Mean error, σ=1.5px landmark noise | Worst case (200 trials) |
  |---|---|---|---|
  | 0°  | 0.00° | 0.92° | 2.60° |
  | 5°  | 0.00° | 0.48° | 2.07° |
  | 10° | 0.00° | 0.44° | 2.28° |
  | 20° | 0.00° | 0.46° | 2.46° |
  | 30° | 0.00° | 0.42° | 2.37° |

  (averaged across yaw/pitch/roll axes; full per-axis breakdown in the
  script's output). Mean error across all 3,000 trials: **0.54°**. Worst
  single-trial error: **2.60°**. The noise-free row proves the solvePnP +
  calibration math is exact; the noisy rows are a sensitivity analysis under
  a stated, not measured, pixel-jitter assumption — real MediaPipe
  localization error wasn't independently characterized against ground
  truth (would need real annotated photos, not synthetic ones).

  Known limitation, found during testing: the Haar cascade face detector
  loses the face entirely past ~20° rotation — a real constraint of that
  detector, not of the pose math. A production system would swap in
  MediaPipe's own face detector (more rotation-robust) ahead of the mesh
  step.
- **Auth**: real JWT issuance/verification (PyJWT, HS256), all
  session-mutating endpoints require a valid bearer token.
- **Persistence**: SQLite (WAL mode) — anomaly history and session state
  survive a process restart, unlike an in-memory dict.
- **Tests**: 19 passing pytest cases (up from an earlier, non-reproducible
  count — 4 tests that need a real face photo now skip cleanly on a fresh
  clone instead of erroring; see `tests/fixtures/README.md`) covering pose
  math (synthetic, no photo needed), signature-comparison math (synthetic),
  face detection, JWT issuance/expiry/tamper-rejection, and SQLite
  persistence.

**Explicitly not production-grade (by design, for portfolio scope):**
- Identity verification uses a lightweight intensity-histogram signature,
  not a trained face-embedding model. This is the one component I'd
  prioritize replacing first — swap in FaceNet/ArcFace/InsightFace behind
  `_face_signature()` / `compare_signatures()` in `vision.py`; the rest of
  the pipeline doesn't need to change.
- Single-instance SQLite, not a shared DB — fine for local dev/demo, not for
  multi-worker horizontal scaling.

## API
| Method | Endpoint                          | Auth | Description                        |
|--------|------------------------------------|------|--------------------------------------|
| GET    | `/health`                          | no   | Liveness check                      |
| POST   | `/auth/login`                      | no   | Issue a JWT (demo: no password check yet) |
| POST   | `/sessions/<id>/register`          | yes  | Register reference face (multipart) |
| POST   | `/sessions/<id>/analyze`           | yes  | Analyze one frame (multipart)       |
| GET    | `/sessions/<id>/summary`           | yes  | Aggregate anomaly summary           |

## Setup
```bash
pip install -r requirements.txt
cd src
python app.py        # serves on http://localhost:5000
```

Or with Docker:
```bash
docker build -t exam-monitor .
docker run -p 5000:5000 -e JWT_SECRET=your-secret exam-monitor
```

## Try it
Needs any real face photo of your own as `face.jpg` in your working directory
(not included in the repo — see `tests/fixtures/README.md` for why).
```bash
TOKEN=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"proctor_id":"proctor_1"}' http://localhost:5000/auth/login \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -F "image=@face.jpg" -H "Authorization: Bearer $TOKEN" \
  http://localhost:5000/sessions/demo1/register

curl -F "image=@face.jpg" -H "Authorization: Bearer $TOKEN" \
  http://localhost:5000/sessions/demo1/analyze

curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/sessions/demo1/summary
```

## Tests
```bash
pip install -r requirements.txt
pytest tests/ -v      # 19 passing, 4 skipped (need a local face photo -- see tests/fixtures/README.md)
```

## Tech Stack
Python · OpenCV · MediaPipe · Flask · PyJWT · SQLite · REST APIs · Docker
