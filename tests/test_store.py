import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import importlib


def _fresh_store(tmp_path):
    """Reload store.py pointed at a temp DB so tests don't touch the real one."""
    import store as store_module
    store_module.DB_PATH = tmp_path / "test_sessions.db"
    importlib.reload(store_module)
    store_module.DB_PATH = tmp_path / "test_sessions.db"
    store_module.init_db()
    return store_module


def test_session_persists_reference(tmp_path):
    store = _fresh_store(tmp_path)
    store.get_or_create_session("s1")
    store.set_reference("s1", b"fake-signature-bytes")
    assert store.get_reference("s1") == b"fake-signature-bytes"


def test_record_frame_and_summary(tmp_path):
    store = _fresh_store(tmp_path)
    store.get_or_create_session("s2")
    store.record_frame("s2", ["no_face_detected"])
    store.record_frame("s2", [])
    store.record_frame("s2", ["no_face_detected", "identity_mismatch"])

    summary = store.get_summary("s2")
    assert summary["frames_analyzed"] == 3
    assert summary["anomaly_counts"]["no_face_detected"] == 2
    assert summary["anomaly_counts"]["identity_mismatch"] == 1
