import cv2
import numpy as np


def draw_hud(img, title, color=(0, 255, 0)):
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.putText(img, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def process_multi_view_frame(frame, frame_idx, graph_res):
    h, w, _ = frame.shape
    ev = graph_res.get("event_trigger", {})
    det = ev.get("details", {})
    hy = ev.get("hysteresis", {})
    vp = graph_res.get("vlm_payload", {})
    plan = graph_res.get("plan", {})
    va_ma = float(det.get("vision_anomaly_ma", 0.0))
    is_trig = bool(ev.get("is_triggered", False))
    hy_s = str(hy.get("state", "idle")).upper()
    vp_m = str(vp.get("mode", "symbolic")).upper()
    pm = str(plan.get("mode", "auto_sim")).upper()

    v1 = draw_hud(frame.copy(), "[Baseline] Standard Detection", (200, 200, 200))
    cv2.rectangle(v1, (w // 4, h // 4), (w // 2, h // 2), (255, 255, 0), 2)

    c_a = (0, 0, 255) if va_ma > 0.7 else (0, 255, 0)
    v2 = draw_hud(frame.copy(), f"[Exp A] S_A AnomMA:{va_ma:.2f}", c_a)

    c_b = (0, 0, 255) if is_trig else (0, 255, 0)
    v3 = draw_hud(frame.copy(), f"[Exp B] S_B HYST:{hy_s} TRIG:{is_trig}", c_b)

    c_c = (0, 0, 255) if pm == "fallback" else (0, 255, 0)
    v4 = draw_hud(frame.copy(), f"[Exp C] S_C PLAN:{pm} VLM:{vp_m}", c_c)

    top = np.hstack((v1, v2))
    bot = np.hstack((v3, v4))
    return np.vstack((top, bot))
