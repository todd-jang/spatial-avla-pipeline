"""Acceptance tests for todo 16a (DoA readout on the ExpB/ExpC HUD tile)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app.multiview import (  # noqa: E402
    ALERT, DOA_ERR_WARN_DEG, GT_GREEN, TILE_H, TILE_W, _doa_strip,
    process_multi_view_frame,
)


def base_state(**extra):
    s = {"event_trigger": {"details": {}, "hysteresis": {}},
         "vlm_payload": {}, "plan": {}, "meta_exec": {}}
    s.update(extra)
    return s


def tile_with(doa):
    return _doa_strip(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8), doa)


def strip_green_pixels(tile):
    """Count pixels matching the green or the alert colour inside the DoA strip band."""
    band = tile[36:61]
    green = np.all(band == np.array(GT_GREEN), axis=-1).sum()
    red = np.all(band == np.array(ALERT), axis=-1).sum()
    return int(green), int(red)


def drawn_rows(tile):
    """Row indices that differ from an all-zero tile, i.e. the true drawn extent."""
    return np.flatnonzero(tile.any(axis=(1, 2)))


DOA_STRIP_ROWS = (35, 61)


def test_geometry_is_unchanged():
    out = process_multi_view_frame(np.zeros((540, 960, 3), dtype=np.uint8), 0,
                                   base_state(doa={"dir_deg": 28.4, "gt_dir_deg": 30.0,
                                                   "confidence": 0.81, "direction": "right"}))
    assert out.shape == (1080, 1920, 3), out.shape


def test_missing_gt_renders_without_error():
    out = process_multi_view_frame(np.zeros((540, 960, 3), dtype=np.uint8), 0,
                                   base_state(doa={"dir_deg": -12.0, "gt_dir_deg": -1.0,
                                                   "confidence": 0.4, "direction": "left"}))
    assert out.shape == (1080, 1920, 3)
    green, _ = strip_green_pixels(out[540:, 960:])
    assert green > 0, "gt=-1 must not be treated as an error state"


def test_accurate_estimate_is_green():
    _, red = strip_green_pixels(tile_with({"dir_deg": 28.4, "gt_dir_deg": 30.0,
                                           "confidence": 0.81}))
    assert red == 0


def test_error_over_threshold_is_red():
    # Literals, never DOA_ERR_WARN_DEG: deriving the stimulus from the constant under
    # test moves code and input together and makes the assertion unable to fail.
    tile = tile_with({"dir_deg": 0.0, "gt_dir_deg": 20.0, "confidence": 0.9})
    _, red = strip_green_pixels(tile)
    assert red > 0, "|err| = 20 deg must turn the readout red"


def test_error_just_under_threshold_stays_green():
    tile = tile_with({"dir_deg": 0.0, "gt_dir_deg": 10.0, "confidence": 0.9})
    _, red = strip_green_pixels(tile)
    assert red == 0, "|err| = 10 deg must stay green, so the boundary sits above it"


def test_ambiguous_is_red_even_when_error_is_small():
    tile = tile_with({"dir_deg": 10.0, "gt_dir_deg": 10.0, "confidence": 0.9,
                      "ambiguous": True})
    _, red = strip_green_pixels(tile)
    assert red > 0, "an ambiguous estimate must turn the readout red"


def test_no_doa_leaves_tile_untouched():
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    out = _doa_strip(tile, {})
    assert np.array_equal(out, tile)


def test_readout_sits_below_the_hud_bar_not_as_a_new_panel():
    tile = tile_with({"dir_deg": 5.0, "gt_dir_deg": 5.0, "confidence": 0.5})
    lo, hi = DOA_STRIP_ROWS
    rows = drawn_rows(tile)
    assert rows.min() == lo, f"strip must start at row {lo}, starts at {rows.min()}"
    assert rows.max() == hi, f"strip must end at row {hi}, ends at {rows.max()}"
    assert not tile[hi + 1:].any(), "nothing may be drawn below the DoA strip"


def test_strip_does_not_overlap_the_hud_bar():
    tile = tile_with({"dir_deg": 5.0, "gt_dir_deg": 5.0, "confidence": 0.5})
    assert not tile[:35].any(), "_doa_strip must not paint over the HUD bar rows"