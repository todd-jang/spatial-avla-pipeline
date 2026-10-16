import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.app.gradio_app import demo

if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() == "true"
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=share,
    )
