import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import time
import jwt
import pytest
from auth import issue_token, SECRET_KEY


def test_issue_token_roundtrip():
    token = issue_token("proctor_42")
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    assert payload["sub"] == "proctor_42"
    assert payload["exp"] > time.time()


def test_expired_token_rejected():
    payload = {"sub": "proctor_1", "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])


def test_tampered_token_rejected():
    token = issue_token("proctor_1")
    tampered = token[:-2] + "xx"
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(tampered, SECRET_KEY, algorithms=["HS256"])
