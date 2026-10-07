"""Tests for scripts/fetch_real_sirens.py acquisition guards.

The fetcher only stores audio it can trace to a permissive licence. archive.org
nests licence fields under the item's `metadata` key, not at the response root;
the guard must read the nested location or every CC0 item is refused (the bug
that blocked the first run) and must refuse items that declare no licence.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from scripts import fetch_real_sirens as frs  # noqa: E402


def test_nested_cc0_licenseurl_is_accepted():
    meta = {"metadata": {"licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/"}}
    assert frs.licence_of(meta, "x.wav")[0] == "CC0-1.0"


def test_missing_license_is_unknown_not_cc0():
    meta = {"metadata": {}}
    assert frs.licence_of(meta, "x.wav")[0] == "unknown"


def test_siren_name_hints_select_ambulance_and_emergency():
    names = [
        {"name": "G44-04-Ambulance Siren.wav", "source": "original", "size": "100"},
        {"name": "notes.txt", "source": "original", "size": "10"},
        {"name": "G44-07-Emergency Siren.wav", "source": "original", "size": "100"},
    ]
    picked = [f["name"] for f in frs.pick_files(names)]
    assert picked == ["G44-04-Ambulance Siren.wav", "G44-07-Emergency Siren.wav"]


def test_original_wav_outranks_derived_mp3_of_same_stem():
    names = [
        {"name": "ambulance.wav", "source": "original", "size": "100"},
        {"name": "ambulance.mp3", "source": "derivative", "size": "50"},
    ]
    picked = frs.pick_files(names)
    assert len(picked) == 1 and picked[0]["name"] == "ambulance.wav"