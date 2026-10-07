"""Acceptance tests for todo 10 (stereo ingest + typed doa label)."""
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.agents.s_i_init import s_i_init_fn  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

SCHEMA = json.loads((ROOT / "contracts/frame_audio_packet_v1.0.json").read_text())
DOA_SCHEMA = SCHEMA["properties"]["doa"]
DOA_KEYS = {"dir_deg", "gt_dir_deg", "confidence", "direction"}
TOP_REQUIRED = ["frame_idx", "timestamp_ms", "audio", "vlm_payload",
                "event_trigger", "meta_exec"]

HYSTERESIS = {"min_hold_frames": 35, "state": "idle", "active_since_frame_idx": -1,
              "active_until_frame_idx": -1, "last_trigger_frame_idx": -1}


def base_state(**extra):
    """Realistic packet skeleton: s_b uses setdefault, so hysteresis must be complete."""
    state = {"frame_idx": 3,
             "event_trigger": {"details": {}, "hysteresis": dict(HYSTERESIS)}}
    state.update(extra)
    return state


def test_required_list_is_unchanged():
    assert SCHEMA["required"] == TOP_REQUIRED


def test_no_new_top_level_key_was_added():
    assert "doa" not in TOP_REQUIRED


def test_doa_subschema_requires_all_four_keys():
    assert set(DOA_SCHEMA["required"]) == DOA_KEYS


def test_mock_validates_against_contract_and_subschema():
    mock = json.loads((ROOT / "mocks/mock_base.json").read_text())
    jsonschema.validate(mock, SCHEMA)
    jsonschema.validate(mock["doa"], DOA_SCHEMA)
    jsonschema.validate(mock["audio"]["doa"], DOA_SCHEMA)


def test_both_mocks_agree_on_doa_shape():
    from mocks.mock_base import MOCK_STATE

    py = json.loads(json.dumps(MOCK_STATE["doa"]))
    js = json.loads((ROOT / "mocks/mock_base.json").read_text())["doa"]
    assert py == js
    assert set(py) == DOA_KEYS
    assert json.loads(json.dumps(MOCK_STATE["audio"]["doa"])) == \
        json.loads((ROOT / "mocks/mock_base.json").read_text())["audio"]["doa"]


def test_subschema_rejects_missing_gt_dir_deg():
    bad = {"dir_deg": 12.0, "confidence": 0.5, "direction": "right"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, DOA_SCHEMA)


def test_subschema_rejects_non_numeric_dir_deg():
    bad = {"dir_deg": "12", "gt_dir_deg": 12.0, "confidence": 0.5, "direction": "right"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, DOA_SCHEMA)


def test_subschema_rejects_confidence_out_of_range():
    bad = {"dir_deg": 1.0, "gt_dir_deg": 1.0, "confidence": 1.5, "direction": "right"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, DOA_SCHEMA)


def test_subschema_rejects_unknown_direction():
    bad = {"dir_deg": 1.0, "gt_dir_deg": 1.0, "confidence": 0.5, "direction": "behind"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, DOA_SCHEMA)


def test_gt_dir_deg_minus_one_is_accepted_as_unknown():
    jsonschema.validate(
        {"dir_deg": 0.0, "gt_dir_deg": -1.0, "confidence": 0.0, "direction": "center"},
        DOA_SCHEMA)


def test_s_i_init_has_all_four_doa_keys():
    state = s_i_init_fn({})
    assert set(state["doa"]) == DOA_KEYS
    assert set(state["audio"]["doa"]) == DOA_KEYS
    assert state["doa"]["gt_dir_deg"] == -1.0
    jsonschema.validate(state["doa"], DOA_SCHEMA)


def test_s_i_init_doa_defaults_validate():
    jsonschema.validate(s_i_init_fn({})["doa"], DOA_SCHEMA)


@pytest.mark.parametrize("source", ["doa", "audio.doa"])
def test_agent_s_b_preserves_incoming_dir_deg(source):
    incoming = {"dir_deg": 42.5, "gt_dir_deg": 45.0,
                "confidence": 0.87, "direction": "right"}
    state = base_state()
    if source == "doa":
        state["doa"] = dict(incoming)
    else:
        state["audio"] = {"doa": dict(incoming)}
    out = s_b_fn(state)
    assert out["audio"]["doa"]["dir_deg"] == 42.5
    assert out["audio"]["doa"]["gt_dir_deg"] == 45.0
    assert out["audio"]["doa"]["confidence"] == 0.87
    assert out["audio"]["doa"]["direction"] == "right"
    assert set(out["audio"]["doa"]) == DOA_KEYS


def test_agent_s_b_defaults_when_no_doa_arrives():
    out = s_b_fn(base_state(frame_idx=0))
    assert set(out["audio"]["doa"]) == DOA_KEYS
    assert out["audio"]["doa"]["dir_deg"] == 0.0
    assert out["audio"]["doa"]["gt_dir_deg"] == -1.0


@pytest.mark.parametrize("source", ["doa", "audio.doa"])
def test_agent_s_b_leaves_incoming_doa_untouched(source):
    """Todo 10 rebuilds audio.doa, so the incoming label must not be written through.

    Scoped to the doa subtree on purpose: s_b writes into the shared nested
    event_trigger dict throughout this codebase, and changing that is out of scope.
    """
    incoming = {"dir_deg": 42.5, "gt_dir_deg": 45.0,
                "confidence": 0.87, "direction": "right"}
    state = base_state()
    if source == "doa":
        state["doa"] = dict(incoming)
    else:
        state["audio"] = {"doa": dict(incoming)}
    s_b_fn(state)
    live = state["doa"] if source == "doa" else state["audio"]["doa"]
    assert live == incoming


def test_fixture_frames_declare_unknown_ground_truth():
    frames = sorted((ROOT / "golden/fixtures/seq_lead_lost_15f").glob("frame_*.json"))
    assert len(frames) == 15
    for f in frames:
        frame = json.loads(f.read_text())
        doa = frame["audio"]["doa"]
        assert set(doa) == DOA_KEYS
        assert doa["gt_dir_deg"] == -1.0


def test_precache_never_feeds_stereo_to_wavlm():
    src = (ROOT / "scripts/precache_wavlm.py").read_text()
    assert "load_wav_stereo_16k" in src
    assert "load_wav_mono_16k" not in src
    assert "stereo.shape[1] == 2" in src
    assert "stereo.mean(axis=1)" in src
