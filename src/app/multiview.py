"""2x2 stitched comparison view.

Layout: TL=GT, TR=Baseline, BL=ExpA AnomMA, BR=ExpB HYST/TRIG + ExpC PLAN/VLM.
Tiles are fixed at 960x540 so the output is always 1920x1080 and never 4K.
"""
import cv2
import numpy as np

from src.app.mpc_simulation import compute_fixed_demo_mpc

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
    raw_conf = doa.get("confidence")
    conf = float(raw_conf or 0.0)
    estimate_available = conf > 0.0
    est = float(doa.get("dir_deg", 0.0) or 0.0) if estimate_available else None
    ambiguous = bool(doa.get("ambiguous", False))
    raw_gt = doa.get("gt_dir_deg", -1.0)
    gt_txt, err = "n/a", None
    if raw_gt is not None and float(raw_gt) != -1.0:
        gt_val = float(raw_gt)
        gt_txt = f"{gt_val:.1f}"
        if estimate_available:
            err = abs(est - gt_val)
    if not estimate_available and gt_txt == "n/a":
        text = "DOA unavailable (no live audio input)"
    else:
        est_txt = f"{est:.1f}" if estimate_available else "n/a"
        text = f"DOA gt={gt_txt} est={est_txt}"
    if err is not None:
        text += f" err={err:.1f}"
    conf_text = f"{conf:.2f}" if estimate_available else "n/a"
    text += f" conf={conf_text}" + (" AMBIGUOUS" if ambiguous else "")
    bad = ambiguous or (err is not None and err > DOA_ERR_WARN_DEG)
    cv2.rectangle(tile, (0, 35), (tile.shape[1], 61), (20, 20, 20), -1)
    cv2.putText(tile, text, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                ALERT if bad else GT_GREEN, 1)
    return tile


def _score_text(details: dict, key: str, precision: int = 3) -> str:
    value = details.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _detection_text(value) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _score_strip(tile, lines):
    top, bottom = 64, 116
    cv2.rectangle(tile, (0, top), (tile.shape[1], bottom), (20, 20, 20), -1)
    for index, text in enumerate(lines):
        cv2.putText(tile, text, (10, 85 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    return tile


def _number(value, precision=3, suffix=""):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}{suffix}"


def _perception_panel(tile, frame, frame_idx, graph_res):
    ev = graph_res.get("event_trigger") or {}
    details = ev.get("details") or {}
    hysteresis = ev.get("hysteresis") or {}
    meta = graph_res.get("meta_exec") or {}
    doa = graph_res.get("doa") or (graph_res.get("audio") or {}).get("doa") or {}
    doa_available = bool(
        (doa.get("confidence") is not None and float(doa["confidence"]) > 0.0)
        or (
            doa.get("gt_dir_deg") is not None
            and float(doa["gt_dir_deg"]) != -1.0
        )
    )
    boxes = (graph_res.get("percept") or {}).get("boxes") or []
    plan = graph_res.get("plan") or {}
    vlm = graph_res.get("vlm_payload") or {}
    score_status = meta.get("score_status") or {}
    mpc = graph_res.get("mpc_simulation") or compute_fixed_demo_mpc()
    mpc_scenario = mpc["scenario"]

    confidences = []
    labels = []
    areas = []
    src_h, src_w = frame.shape[:2]
    for item in boxes:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("class") or item.get("class_name")
        if label:
            labels.append(str(label))
        confidence = item.get("confidence", item.get("conf"))
        if confidence is not None:
            confidences.append(float(confidence))
        bbox = item.get("bbox") or item.get("box")
        if bbox and len(bbox) >= 4 and src_h and src_w:
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            areas.append(
                max(0.0, x2 - x1) * max(0.0, y2 - y1) / (src_h * src_w)
            )

    remaining = max(
        0,
        int(hysteresis.get("active_until_frame_idx", -1))
        - int(graph_res.get("frame_idx", frame_idx)),
    )
    audio = graph_res.get("audio") or {}
    siren = audio.get("siren") or {}
    siren_score = details.get("p_siren", siren.get("confidence"))
    if score_status.get("siren") in ("report_unavailable", "not_connected"):
        siren_score = None
    vision_unavailable = score_status.get("rtdetr") in (
        "detections_unavailable",
        "not_connected",
    )
    if vision_unavailable:
        risk_score = None
        risk_ma = None
    else:
        risk_score = details.get("vision_anomaly_score")
        risk_ma = details.get("vision_anomaly_ma")
    if not boxes:
        boxes_count = "n/a"
        box_conf_max = "n/a"
        box_conf_mean = "n/a"
        box_area_max = "n/a"
    else:
        boxes_count = str(len(boxes))
        box_conf_max = _number(max(confidences), 2) if confidences else "n/a"
        box_conf_mean = (
            _number(sum(confidences) / len(confidences), 2)
            if confidences
            else "n/a"
        )
        box_area_max = (
            _number(max(areas) * 100, 2, "%") if areas else "n/a"
        )
    trigger_status = (
        f"raw={bool(ev.get('raw_trigger', False))} "
        f"active={bool(ev.get('is_triggered', False))} "
        f"hyst={str(hysteresis.get('state', 'n/a')).upper()} hold={remaining}f"
    )
    if vision_unavailable and siren_score is None:
        trigger_status = "n/a (no measured live perception input)"
    doa_direction = doa.get("direction", "n/a")
    doa_text = "n/a (not connected)"
    if doa_available:
        doa_text = (
            f"{_number(doa.get('dir_deg'), 1)}deg/"
            f"{_number(doa.get('confidence'), 2)} {doa_direction}"
        )
    gt_direction = doa.get("gt_dir_deg")
    doa_gt = "n/a"
    if doa_available and gt_direction is not None and float(gt_direction) != -1.0:
        doa_gt = (
            f"{_number(gt_direction, 1)}deg err="
            f"{_number(abs(float(doa.get('dir_deg', 0.0)) - float(gt_direction)), 1)}deg"
        )

    qwen_source = vlm.get("llm_source", "n/a")
    qwen_text = vlm.get("llm_action_text", "n/a")
    qwen_confidence = vlm.get("confidence")
    lat_ms = meta.get("lat_ms")
    lines = [
        f"FRAME {frame_idx} | DETECTIONS {boxes_count} | "
        f"LEAD_LOST {details.get('lead_lost', 'n/a') if not vision_unavailable else 'n/a'} | "
        f"LANE_LOST {details.get('lane_lost_count', 'n/a') if not vision_unavailable else 'n/a'}",
        f"BOX CONF max/mean {box_conf_max}/{box_conf_mean} | "
        f"BOX AREA max {box_area_max}",
        f"VISION risk {_number(risk_score)} MA5 {_number(risk_ma)} | "
        f"MAE {_number(details.get('vision_anomaly_mae'), 4)} "
        f"MA5 {_number(details.get('vision_anomaly_mae_ma'), 4)}",
        f"AUDIO p_siren {_number(siren_score)} | "
        f"DOA {doa_text} | GT {doa_gt}",
        f"DOA rate {_number(details.get('doa_change_rate_dps_ma'), 1, 'deg/s') if doa_available else 'n/a'} | "
        f"TRIGGER {trigger_status}",
        f"PLAN (SIM) steer {_number(plan.get('steer_deg'), 1, 'deg')} "
        f"speed {_number(plan.get('speed_kmh'), 1, 'km/h')} | "
        f"QWEN {qwen_source}: {qwen_text[:48]} "
        f"conf={_number(qwen_confidence, 2)}",
        f"MPC DEMO steer {_number(mpc.get('steering_angle_degrees'), 1, 'deg')} | "
        f"brake {_number(mpc.get('brake_percent'), 0, '%')} "
        f"lamp {'ON' if mpc.get('brake_lamp_on') else 'OFF'} | "
        f"throttle {_number(mpc.get('throttle_percent'), 0, '%')}",
        f"MPC VIRTUAL inputs: lat {mpc_scenario['lateral_error_m']:.1f}m "
        f"heading {mpc_scenario['heading_error_deg']:.1f}deg "
        f"speed {mpc_scenario['speed_kmh']:.0f}km/h "
        f"TTC {mpc_scenario['obstacle_ttc_s']:.1f}s | "
        f"predicted lateral {mpc['predicted_final_lateral_error_m']:.2f}m",
        f"GRAPH latency {_number(lat_ms, 1, 'ms')} | "
        f"graph FPS {_number(1000.0 / float(lat_ms), 1) if lat_ms and float(lat_ms) > 0 else 'n/a'}",
    ]
    top, row_height, padding = 122, 21, 7
    bottom = top + padding * 2 + row_height * len(lines)
    cv2.rectangle(tile, (0, top), (tile.shape[1], bottom), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            tile,
            line,
            (10, top + padding + 17 + index * row_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )


def process_multi_view_frame(frame, frame_idx, graph_res):
    graph_res = dict(graph_res)
    graph_res.setdefault("mpc_simulation", compute_fixed_demo_mpc())
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

    t_a = draw_hud(
        _fit(frame),
        f"[Vision] RT-DETR:{_score_text(det, 'vision_anomaly_score')} "
        f"MA5:{_score_text(det, 'vision_anomaly_ma', 2)}",
        ALERT if va_ma > 0.7 else GT_GREEN,
    )
    _score_strip(t_a, [
        f"MAE-Tiny distance: {_score_text(det, 'vision_anomaly_mae', 4)}",
        f"MAE-Tiny MA5: {_score_text(det, 'vision_anomaly_mae_ma', 4)}",
    ])

    t_b = draw_hud(
        _fit(frame),
        f"[Siren] p:{_score_text(det, 'p_siren')} "
        f"AST:{_score_text(det, 'siren_score_mae', 4) if 'siren_score_mae' in det else _score_text(det, 'siren_score_template')} "
        f"MAE:{_detection_text(det.get('siren_detected_mae'))} "
        f"HYST:{hy_s} TRIG:{is_trig}",
        ALERT if is_trig else GT_GREEN,
    )
    _doa_strip(t_b, graph_res.get("doa") or (graph_res.get("audio") or {}).get("doa") or {})
    _score_strip(t_b, [
        f"p_siren MA5: {_score_text(det, 'p_siren_ma5')}",
        f"p_siren delta: {_score_text(det, 'p_siren_delta')}",
        f"AST strength: {det.get('siren_strength_mae') or 'n/a'}",
    ])

    cv2.putText(
        t_b,
        "Audio classifier: offline scores only",
        (10, TILE_H - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        PASSIVE,
        1,
    )

    for tile in (t_gt, t_base, t_a, t_b):
        _perception_panel(tile, frame, frame_idx, graph_res)

    top = np.hstack((t_gt, t_base))
    bot = np.hstack((t_a, t_b))
    return np.vstack((top, bot))
