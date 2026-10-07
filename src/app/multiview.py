"""2x2 stitched comparison view.

Layout: TL=GT, TR=Baseline, BL=ExpA AnomMA, BR=ExpB HYST/TRIG + ExpC PLAN/VLM.
Tiles are fixed at 960x540 so the output is always 1920x1080 and never 4K.
"""
import cv2
import numpy as np

TILE_W, TILE_H = 960, 540

GT_GREEN = (0, 255, 0)
VEHICLE_BLUE = (255, 128, 0)
PEDESTRIAN_RED = (0, 0, 255)
PASSIVE = (200, 200, 200)
ALERT = (0, 0, 255)

DOA_ERR_WARN_DEG = 15.0


def draw_hud(img, title, color=GT_GREEN):
    cv2.rectangle(img, (0, 0), (img.shape[1], 35), (20, 20, 20), -1)
    cv2.putText(img, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def _fit(frame):
    interp = cv2.INTER_AREA if frame.shape[0] > TILE_H else cv2.INTER_LINEAR
    return cv2.resize(frame, (TILE_W, TILE_H), interpolation=interp)


def _iter_boxes(raw, src_w, src_h):
    """Accept [x1,y1,x2,y2(,label)] or {bbox|box:[...], label|class:str}.

    percept.boxes is [] everywhere in the codebase so far (only s_i_init
    setdefaults it), so this stays tolerant instead of assuming one shape.
    """
    sx, sy = TILE_W / float(src_w), TILE_H / float(src_h)
    for item in raw or ():
        if isinstance(item, dict):
            coords = item.get("bbox") or item.get("box")
            label = item.get("label") or item.get("class") or "vehicle"
        else:
            coords = item
            label = "vehicle"
        if not coords or len(coords) < 4:
            continue
        x1, y1, x2, y2 = (int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(coords[:4]))
        if len(coords) > 4 and not isinstance(item, dict):
            label = coords[4]
        yield x1, y1, x2, y2, str(label)


def _status_strip(tile, meta):
    parts = [p for p in (meta.get("model_name"),) if p]
    for label, key, fmt in (("FPS", "fps", "{:.1f}"), ("mAP", "mAP", "{:.3f}"), ("LAT", "lat_ms", "{:.1f}ms")):
        if meta.get(key) is not None:
            parts.append(f"{label}:{fmt.format(float(meta[key]))}")
    if not parts:
        return
    cv2.rectangle(tile, (0, TILE_H - 28), (TILE_W, TILE_H), (20, 20, 20), -1)
    cv2.putText(tile, "  ".join(parts), (10, TILE_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _doa_strip(tile, doa):
    """Per-frame DoA readout drawn under the BR HUD bar (no extra panel).

    gt renders as "n/a" when gt_dir_deg == -1, and the readout turns red when
    |err| exceeds DOA_ERR_WARN_DEG or the estimate is flagged ambiguous.
    """
    if not doa:
        return tile
    est = float(doa.get("dir_deg", 0.0) or 0.0)
    conf = float(doa.get("confidence", 0.0) or 0.0)
    ambiguous = bool(doa.get("ambiguous", False))
    raw_gt = doa.get("gt_dir_deg", -1.0)
    gt_txt, err = "n/a", None
    if raw_gt is not None and float(raw_gt) != -1.0:
        gt_val = float(raw_gt)
        gt_txt = f"{gt_val:.1f}"
        err = abs(est - gt_val)
    text = f"DOA gt={gt_txt} est={est:.1f}"
    if err is not None:
        text += f" err={err:.1f}"
    text += f" conf={conf:.2f}" + (" AMBIGUOUS" if ambiguous else "")
    bad = ambiguous or (err is not None and err > DOA_ERR_WARN_DEG)
    cv2.rectangle(tile, (0, 35), (tile.shape[1], 61), (20, 20, 20), -1)
    cv2.putText(tile, text, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                ALERT if bad else GT_GREEN, 1)
    return tile


def process_multi_view_frame(frame, frame_idx, graph_res):
    src_h, src_w = frame.shape[:2]
    ev = graph_res.get("event_trigger", {})
    det = ev.get("details", {})
    hy = ev.get("hysteresis", {})
    vp = graph_res.get("vlm_payload", {})
    plan = graph_res.get("plan", {})
    meta = graph_res.get("meta_exec", {}) or {}

    va_ma = float(det.get("vision_anomaly_ma", 0.0) or 0.0)
    is_trig = bool(ev.get("is_triggered", False))
    hy_s = str(hy.get("state", "idle")).upper()
    vp_m = str(vp.get("mode", "symbolic")).upper()
    pm = str(plan.get("mode", "auto_sim")).upper()

    gt = _fit(frame)
    for x1, y1, x2, y2, label in _iter_boxes((graph_res.get("percept", {}) or {}).get("boxes"), src_w, src_h):
        color = PEDESTRIAN_RED if "ped" in label.lower() else VEHICLE_BLUE
        cv2.rectangle(gt, (x1, y1), (x2, y2), color, 2)
        cv2.putText(gt, label, (x1, max(36, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if not (graph_res.get("percept", {}) or {}).get("has_lead"):
        cv2.rectangle(gt, (TILE_W - 210, TILE_H - 40), (TILE_W - 10, TILE_H - 14), GT_GREEN, 2)
        cv2.putText(gt, "lead lost", (TILE_W - 200, TILE_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_GREEN, 1)
    t_gt = draw_hud(gt, "[GT] Ground Truth", GT_GREEN)
    _status_strip(t_gt, meta)

    t_base = draw_hud(_fit(frame), "[Baseline] Standard Detection", PASSIVE)

    t_a = draw_hud(_fit(frame), f"[Exp A] S_A AnomMA:{va_ma:.2f}", ALERT if va_ma > 0.7 else GT_GREEN)

    t_b = draw_hud(
        _fit(frame),
        f"[Exp B] SIR:{float(det.get('p_siren', 0.0) or 0.0):.2f} "
        f"HYST:{hy_s} TRIG:{is_trig}",
        ALERT if is_trig else GT_GREEN,
    )
    _doa_strip(t_b, graph_res.get("doa") or (graph_res.get("audio") or {}).get("doa") or {})

    # Qwen LLM Text Overlay
    llm_text = vp.get("llm_action_text", "Qwen(Local): WAITING...")
    cv2.putText(t_b, llm_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if is_trig else (200, 200, 200), 2)
    
    # Fake audio waveform visualization
    for i in range(0, TILE_W, 12):
        h = int(np.random.rand() * 25 + 5)
        color = (0, 0, 255) if is_trig else (150, 150, 150)
        cv2.line(t_b, (i, TILE_H - 30 - h), (i, TILE_H - 30 + h), color, 2)

    top = np.hstack((t_gt, t_base))
    bot = np.hstack((t_a, t_b))
    return np.vstack((top, bot))
