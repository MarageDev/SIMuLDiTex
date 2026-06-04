
import sys
import os
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports
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

os.makedirs('./images/results/gif_frames',exist_ok=True)
clear_output()

name1,name2 = 'wall','rust'                 # 2 textures for background and font: 'wall' 'carpet' 'rust' 'crepe' 'ananaskin' 'ananaskin2','gold'
nc = 16                                     # 16, 32    for 1M or 4M parameters
S = 8                                       # Sampling steps
r = .8                                      # renoising time ratio
patch_size=3000                             # Maximum side of patches used if inference triggers memory error, to lower in case this happens.


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

def load_image_tensor(path:str, device:str='cuda', size:tuple=None, scaling_factor:int=None):
    img = Image.open(path).convert('RGB')
    numpy_img = np.array(img)
    h, w, c = numpy_img.shape
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    if scaling_factor is not None:
        img = img.resize((w*scaling_factor,h*scaling_factor), Image.BICUBIC)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return x

omega = load_image_tensor('./Demos/results/test_mask.jpg', device='cuda',scaling_factor=4)
_, c, h, w = omega.shape
size = (h, w)
print(size)
im = diffusion1.spatial_interp(size=size, time_ratio=r, omega=omega, patch_size=patch_size)

plt.figure(figsize=(15, 15))
plt.imshow(im[0].permute(1, 2, 0).cpu())
plt.axis('off')
plt.show()

        
