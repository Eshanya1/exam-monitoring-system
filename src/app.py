"""
Flask REST API for the exam monitoring platform.

Endpoints:
  GET  /health                        - liveness check (no auth)
  POST /auth/login                    - issue a JWT for a proctor_id (demo: no password check)
  POST /sessions/<id>/register        - register reference face (auth required)
  POST /sessions/<id>/analyze         - analyze one webcam frame (auth required)
  GET  /sessions/<id>/summary         - aggregate anomaly summary (auth required)

Session state (reference signature, frame counts, anomaly history) is
persisted to SQLite via store.py, so it survives process restarts.
"""
import io
import time
import logging

import numpy as np
import cv2
from flask import Flask, request, jsonify

from vision import analyze_frame, register_reference
from auth import issue_token, require_auth
import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exam-monitor")

app = Flask(__name__)
store.init_db()


def _read_image(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/auth/login")
def login():
    """Demo login: issues a token for any proctor_id. Wire up real
    credential verification (DB lookup + password hash check, or SSO)
    before using this outside a local/demo environment."""
    body = request.get_json(force=True, silent=True) or {}
    proctor_id = body.get("proctor_id")
    if not proctor_id:
        return jsonify(error="proctor_id required"), 400
    token = issue_token(proctor_id)
    return jsonify(access_token=token, token_type="bearer", expires_in=3600)


@app.post("/sessions/<session_id>/register")
@require_auth
def register(session_id):
    if "image" not in request.files:
        return jsonify(error="image file required"), 400
    image = _read_image(request.files["image"])
    if image is None:
        return jsonify(error="could not decode image"), 400

    signature = register_reference(image)
    if signature is None:
        return jsonify(error="no face detected in reference image"), 422

    store.get_or_create_session(session_id)
    store.set_reference(session_id, signature.tobytes())
    log.info("Registered reference face for session=%s", session_id)
    return jsonify(status="registered", session_id=session_id)


@app.post("/sessions/<session_id>/analyze")
@require_auth
def analyze(session_id):
    if "image" not in request.files:
        return jsonify(error="image file required"), 400
    image = _read_image(request.files["image"])
    if image is None:
        return jsonify(error="could not decode image"), 400

    t0 = time.time()
    ref_bytes = store.get_reference(session_id)
    ref_signature = np.frombuffer(ref_bytes, dtype=np.float32) if ref_bytes else None

    result = analyze_frame(image, reference_signature=ref_signature)
    latency_ms = round((time.time() - t0) * 1000, 1)

    store.record_frame(session_id, result.anomalies)
    if result.anomalies:
        log.warning("session=%s anomalies=%s latency_ms=%s", session_id, result.anomalies, latency_ms)

    return jsonify(
        faces_detected=result.faces_detected,
        identity_match=result.identity_match,
        head_pose_deviation_deg=result.head_pose_deviation_deg,
        anomalies=result.anomalies,
        latency_ms=latency_ms,
    )


@app.get("/sessions/<session_id>/summary")
@require_auth
def summary(session_id):
    data = store.get_summary(session_id)
    return jsonify(session_id=session_id, **data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
