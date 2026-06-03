"""
Notes
Paramètres à exposer sur l'interface :
    - number of diffusion steps T
    - noise ratio r
    - number of sampling steps S
    - patch size
    - scale factors F = {f1, . . . fK}
    
Démos à faire : 
    - Writing text with two textures as font and background (use gradio https://www.gradio.app/docs/gradio/imageeditor to be able to paint over it instead of typing text)
    Maybe add possibility to upload a custom b&w image as a mask too (easier for quick demos)
    - Stylization
    - Spatial linear interpolation between two textures
    - Interpolate two images
    - (Interpolation between textures to make a gif)
    - (Synthesis with a subset of the multi-resolution pyramid)
    - Inference
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
import time
import numpy as np
from Demos.Utilities.theme import *

# SIMuLDiTex Imports
from SIMuLDiTex.SIMuLDiTex import Unet, GaussianDiffusion, Trainer
from torchvision.utils import make_grid,save_image
from PIL import Image
import cv2
import numpy as np
import torch,torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from SIMuLDiTex.ResizeRight import resize
import SIMuLDiTex.interp_methods as interp
from IPython.display import clear_output
import time, json, re



# Functions

## SIMuLDiTex Functions

def n_params(model):
    pp=0
    for p in list(model.parameters(True)):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_latest_model_index(directory):
    files = os.listdir(directory)
    model_files = [f for f in files if re.match(r'model-(\d+)\.pt', f)]
    indices = []
    for model in model_files:
        match = re.match(r'model-(\d+)\.pt', model)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices)
    else:
        return None

def run_simulditex(tex1,tex2):
    """
    tex1 : background texture
    tex2 : the masking texture
    """
    
    
    
    # Texture 1
    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name1,nc)

    model1 = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion1 = GaussianDiffusion(
        model1,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer1 = Trainer(
        diffusion1,
        'images/data/%s'%name1,
        results_folder=folder)
    trainer1.load(get_latest_model_index(folder))

    
    # Texture 2
    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name2,nc)

    model2 = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion2 = GaussianDiffusion(
        model2,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer2 = Trainer(
        diffusion2,
        'images/data/%s'%name2,
        results_folder=folder)
    trainer2.load(get_latest_model_index(folder))


    diffusion1.model2=model2
    

## UI Functions

def predict(im:np.ndarray):
    return im["composite"]

def visu(x):
    comp:np.ndarray = x["composite"]
    print(comp.shape)
    print(comp.flatten())
    print(comp.flatten().reshape(comp.shape))
    
def filter(im):
    comp:np.ndarray = im["composite"].astype(np.float64)
    dim = comp.shape    
    #return comp.astype(np.uint8)#.reshape(dim)
    return comp[:,:,2:4].astype(np.uint8)

## Globals update

def update_textures(tex1,tex2):
    global name1, name2
    name1 = tex1
    name2 = tex2

def update_parameter_number(x):
    global nc
    
    match x:
        case "1M":
            nc = 16
        case "4M":
            nc = 32

def update_simulditex_params(i_s,i_r,i_patch_size):
    global s,r,patch_size
    
    s = i_s
    r = i_r 
    patch_size = i_patch_size        
    
# Interface
# Global variables

name1,name2 = 'wall','rust' # 2 textures for background anbd font: 'wall' 'carpet' 'rust' 'crepe' 'ananaskin' 'ananaskin2','gold'
nc = 16 # 16, 32    for 1M or 4M parameters
S = 2 # Sampling steps
r = .8 # renoising time ratio
patch_size=3000 # Maximum side of patches used if inference triggers memory error, to lower in case this happens.
char_size = 1024 # character size
string='your_text' # use '_' to break line, all lines must have the same number of caracters, you can use blanks ' '
dilation = 2 # iterations of morphological dilation on caracters mask


with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    with gr.Row(equal_height=True,variant="default"):
        in_radio_nc = gr.Radio(
            choices=["1M", "4M"],
            label="Parameters number",
            value="1M",
            elem_classes="radio_group",
        )
    with gr.Row(equal_height=True,variant="panel"):
        im = gr.ImageEditor(
            type="numpy",
            label="Input"
        )
        im_preview = gr.Image(
            type="numpy",
            label="Output"
        )
    with gr.Row(equal_height=True,):
        with gr.Column(scale=1):
            in_drop_tex_1 = gr.Dropdown(
                choices=['wall','carpet','rust','crepe','ananaskin','ananaskin2','gold'],
                label="Texture 1",
                info="Background texture displayed behind the masking texture (Texture 2)",
                value="wall",
                )
            in_drop_tex_2 = gr.Dropdown(
                choices=['wall','carpet','rust','crepe','ananaskin','ananaskin2','gold'],
                label="Texture 2",
                info="Texture displayed above the background texture (Texture 1). The mask input will use this texture.",
                value="rust",
                )
        with gr.Column(scale=2):
            in_S = gr.Slider(
                label="Sampling Steps (S)",
                info="Number of sampling temps",
                value=2,
                minimum=1,
                maximum=20,
                step=1
            )
            in_r = gr.Slider(
                label="Renoising Time Ration (r)",
                info="Renoising time ratio",
                value=0.8,
                minimum=0,
                maximum=1,
                step=0.01
            )
            in_patch_size = gr.Slider(
                label="Patch Size",
                info="Maximum size of patches used (if inference triggers memory error, lower it)",
                value=3000,
                minimum=1,
                maximum=10000,
                step=1
            )
            in_dilation = gr.Slider(
                label="Mask Dilation",
                info="Iterations of morphological dilation on the masking texture (painted texture 2 )",
                value=0,
                minimum=0,
                maximum=10,
                step=1
            )
    
    # Bind events
    ## Bind texture change
    in_drop_tex_1.change(
        fn=update_textures,
        inputs=[in_drop_tex_1, in_drop_tex_2],
        outputs=[]
    )
    in_drop_tex_2.change(
        fn=update_textures,
        inputs=[in_drop_tex_1, in_drop_tex_2],
        outputs=[]
    )
    
    ## Bind paramater count
    in_radio_nc.change(
        fn=update_parameter_number,
        inputs=[in_radio_nc],
        outputs=[]
    )
    
    ## Bind general simulditex parameters
    in_S.change(
        fn=update_simulditex_params,
        inputs=[in_S, in_r, in_patch_size],
        outputs=[]
    )
    in_r.change(
        fn=update_simulditex_params,
        inputs=[in_S, in_r, in_patch_size],
        outputs=[]
    )
    in_patch_size.change(
        fn=update_simulditex_params,
        inputs=[in_S, in_r, in_patch_size],
        outputs=[]
    )
    
    visualize = gr.Button(value="Print to debug", variant="primary")
    apply_filter = gr.Button(value="filter")
    
    apply_filter.click(
        fn=filter,
        inputs=im,
        outputs=im_preview
    )
    visualize.click(visu,im)
    im.change(predict, outputs=im_preview, inputs=im)
    gr.HTML(HTML_FOOTER)


demo.launch(css=CUSTOM_CSS,head=HTML_CUSTOM_HEAD)