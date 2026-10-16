import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app.qwen_action import QwenActionAdvisor  # noqa: E402


def test_advisor_runs_asynchronously_and_deduplicates_context():
    calls = []
    started = threading.Event()
    release = threading.Event()

    def infer(context):
        calls.append(context)
        started.set()
        release.wait(timeout=2)
        return "CAUTION"

    advisor = QwenActionAdvisor(infer_fn=infer)
    assert "loading / analyzing" in advisor.update(False, 0.2, ["car"])
    assert started.wait(timeout=1)
    release.set()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = advisor.status()
        if "CAUTION" in status:
            break
        time.sleep(0.01)

    assert status == "Qwen read-only advisory: CAUTION"
    assert advisor.update(False, 0.2, ["car"]) == status
    assert calls == [
        "Vision risk band 0/4; detections: car; pipeline vision trigger: inactive."
    ]


def test_advisor_surfaces_inference_errors():
    def infer(_context):
        raise ValueError("model unavailable")

    advisor = QwenActionAdvisor(infer_fn=infer)
    advisor.update(False, 0.0, [])

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = advisor.status()
        if "error:" in status:
            break
        time.sleep(0.01)

    assert status == "Qwen advisory error: ValueError: model unavailable"
