import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from Demos.utilities.theme import *
from Demos.sub_demos.spatial_interpolation import demo_spatial_interpolation
from Demos.sub_demos.styilization import demo_stylization

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    with gr.Tabs(selected="a"):
        with gr.Tab("Spatial Interpolation", id="a"):
            demo_spatial_interpolation()
        with gr.Tab("Stylization", id="b"):
            demo_stylization()
    gr.HTML(HTML_FOOTER)


demo.queue().launch(css=CUSTOM_CSS,head=HTML_CUSTOM_HEAD, theme=theme)