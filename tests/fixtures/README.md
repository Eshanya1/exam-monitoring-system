# Test fixtures

`face.jpg` (a clear, frontal photo of a face) is intentionally **not** committed
to this repo — it would mean shipping a real person's photo (or a licensed
stock photo) as test data, which isn't something to commit without clear
rights to redistribute it.

Four tests in `test_vision.py` are integration tests that need this file to
exercise the full Haar-cascade + MediaPipe detection path against a real
photo. They're marked `@requires_fixture` and **skip** (not fail) when the
file is absent, which is what you'll see on a fresh clone.

To run them locally: drop any clear, frontal, well-lit face photo at
`tests/fixtures/face.jpg` and re-run `pytest`.

The deterministic logic these tests exercise doesn't require a photo at all:

- Signature comparison math — `test_signature_comparison_*` in `test_vision.py`
- Pose-estimation math (the thing actually validated for accuracy) —
  `test_pose_synthetic.py` and `scripts/validate_pose_synthetic.py`
