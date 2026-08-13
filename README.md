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
  rotation magnitude** against synthetic ground truth:

  | Applied rotation | Measured deviation |
  |---|---|
  | 0°  | 0.0° |
  | 5°  | 4.9° |
  | 10° | 10.2° |
  | 20° | 19.6° |
  | 30° | 31.0° |

  Known limitation, found during testing: the Haar cascade face detector
  loses the face entirely past ~20° rotation — a real constraint of that
  detector, not of the pose math. A production system would swap in
  MediaPipe's own face detector (more rotation-robust) ahead of the mesh
  step.
- **Auth**: real JWT issuance/verification (PyJWT, HS256), all
  session-mutating endpoints require a valid bearer token.
- **Persistence**: SQLite (WAL mode) — anomaly history and session state
  survive a process restart, unlike an in-memory dict.
- **Tests**: 11 passing pytest cases covering vision logic (face detection,
  identity matching, pose deviation, the documented rotation-limit edge
  case), JWT issuance/expiry/tamper-rejection, and SQLite persistence.

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
pytest tests/ -v      # 11 tests, all passing
```

## Tech Stack
Python · OpenCV · MediaPipe · Flask · PyJWT · SQLite · REST APIs · Docker
