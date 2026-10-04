import cv2
import time
import gradio as gr
from src.core.bridge import GraphBridge
from src.app.multiview import process_multi_view_frame

bridge = GraphBridge(fast_mode=True)


def run(vp):
    if vp is None:
        raise gr.Error("Upload video")
    cap = cv2.VideoCapture(vp)
    frame_idx = 0
    start = time.time() * 1000.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fr = cv2.resize(frame, (480, 270))
        ts = start + frame_idx * 33.333
        grs = bridge.invoke_frame(frame_idx, ts)
        g = process_multi_view_frame(fr, frame_idx, grs)
        lat = grs.get("meta_exec", {}).get("lat_ms", 0.0)
        cv2.putText(g, f"LAT:{lat:.1f}ms", (g.shape[1] - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        yield cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        time.sleep(0.005)
    cap.release()


def build():
    with gr.Blocks(title="Spatial-AVLA 2x2 < 0.33s") as d:
        gr.Markdown("## 2x2 Multi-View (graph_ui Fast Path)")
        with gr.Row():
            vi = gr.Video(label="Input")
            oi = gr.Image(label="Stream", streaming=True, type="numpy")
        gr.Button("Run").click(run, vi, oi, show_progress=False)
    return d


demo = build()
