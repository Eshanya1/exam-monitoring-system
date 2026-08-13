"""
JWT-based auth for the proctoring API. A proctor/invigilator authenticates
once to get a token, then must present it (Authorization: Bearer <token>)
for session-management endpoints. Uses a shared-secret HS256 scheme
suitable for a single-service deployment; swap for RS256 + a shared JWKS
endpoint if this needs to be verified by other services independently.
"""
import os
import time
import functools
import jwt
from flask import request, jsonify

SECRET_KEY = os.environ.get("JWT_SECRET", "dev-secret-change-me")
TOKEN_TTL_SECONDS = 3600


def issue_token(proctor_id: str) -> str:
    payload = {"sub": proctor_id, "iat": int(time.time()), "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def _extract_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header.split(" ", 1)[1]
    return None


def require_auth(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _extract_token()
        if not token:
            return jsonify(error="missing bearer token"), 401
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify(error="token expired"), 401
        except jwt.InvalidTokenError:
            return jsonify(error="invalid token"), 401
        return fn(*args, **kwargs)
    return wrapper
