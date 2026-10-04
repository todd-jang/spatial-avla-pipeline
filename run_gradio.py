import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.app.gradio_app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
