"""Tests for the yt-dlp cookie passthrough in scripts/slice_clips.py.

Todo 15 is blocked on four clips that YouTube's bot gate refuses without an
authenticated session. The slicer must (a) pass --cookies through when given a
file and (b) refuse to attempt an unauthenticated download it cannot possibly
satisfy, instead of silently producing nothing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from scripts import slice_clips  # noqa: E402


def _clip():
    return {"clip_id": "C03", "youtube_id": "GN_g-RsUM94",
            "source_url": "https://www.youtube.com/watch?v=GN_g-RsUM94"}


def test_missing_cookies_file_aborts_before_yt_dlp(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("yt-dlp must not run when the cookies file is absent")

    monkeypatch.setattr(slice_clips, "run", explode)
    monkeypatch.setattr(slice_clips, "free_gib", lambda p: 99.0)
    with pytest.raises(SystemExit) as exc:
        slice_clips.download(_clip(), tmp_path, 1080, 1.5, tmp_path / "absent.txt")
    assert "NO_COOKIES" in str(exc.value)


def test_cookies_flag_is_forwarded_to_yt_dlp(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        return type("R", (), {"returncode": 1, "stderr": "blocked"})()

    monkeypatch.setattr(slice_clips, "run", fake_run)
    monkeypatch.setattr(slice_clips, "free_gib", lambda p: 99.0)
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    assert slice_clips.download(_clip(), tmp_path, 1080, 1.5, cookies) is None
    cmd = seen["cmd"]
    assert "--cookies" in cmd
    assert cmd[cmd.index("--cookies") + 1] == str(cookies)
    assert cmd[-1] == _clip()["source_url"]


def test_no_cookies_flag_means_no_cookies_argument(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        return type("R", (), {"returncode": 1, "stderr": "blocked"})()

    monkeypatch.setattr(slice_clips, "run", fake_run)
    monkeypatch.setattr(slice_clips, "free_gib", lambda p: 99.0)
    slice_clips.download(_clip(), tmp_path, 1080, 1.5, None)
    assert "--cookies" not in seen["cmd"]