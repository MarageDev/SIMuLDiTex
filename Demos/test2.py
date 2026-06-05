import gradio as gr
import numpy as np

with gr.Blocks() as demo:
    W = gr.Slider(label="W", minimum=50, maximum=1000, value=100)
    H = gr.Slider(label="H", minimum=50, maximum=1000, value=500)

    # initial canvas matching slider defaults
    width0 = 100
    height0 = 500

    @gr.render(inputs=[W, H])
    def render_canvas(w, h):
        canvas = {
            "background": None,
            "layers": [np.full((h, w), 255, dtype=np.uint8)],
            "composite": None,
        }
        gr.ImageEditor(type="numpy", label="Canvas", canvas_size=(w, h), value=canvas, interactive=True)

    demo.launch()