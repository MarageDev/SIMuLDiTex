# Imports
import gradio as gr
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
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
import time, os, json, re

def stylize(pth):

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

    name = 'wall'                               # texture source
    content_pth = './images/iccv_honolulu.jpg'  # path to content image
    zeta = 0.5                                 # controls the strengh of the data fidelity
    res = 1                                     # controls the relative size of texture patterns in the output image. Higher value will give an impression of zoom out of the texure. 
    nc = 16                                     # 16, 32    for 1M or 4M parameters
    S = 5                                       # Sampling steps
    r = .7                                      # renoising time ratio
    patch_size = 4096                           # Maximum size of patches used if inference triggers memory error, to lower in case this happens.
    zoom_factor = 1

    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name,nc)

    model = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer = Trainer(
        diffusion,
        'images/data/%s'%name,
        results_folder=folder)
    trainer.load(get_latest_model_index(folder))

    gt=trainer.ds.images[0][0].unsqueeze(0)
    print(pth)
    input=transforms.ToTensor()(Image.open(pth)).unsqueeze(0)


    for im in diffusion.stylize(size=(input.shape[-2]*zoom_factor,input.shape[-1]*zoom_factor),input=input,gt=gt,zeta=.1,output_texture_resolution_ind=res,time_ratio=r):
        img = im[0].permute(1,2,0).cpu()
        img = (img * 255).clip(0, 255).numpy().astype(np.uint8)
        yield img

demo = gr.Interface(
    stylize,
    gr.Image(type="filepath"),
    gr.Image(type="numpy")
)
demo.queue().launch()