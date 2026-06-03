
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

name1,name2 = 'wall','rust'                 # 2 textures for background anbd font: 'wall' 'carpet' 'rust' 'crepe' 'ananaskin' 'ananaskin2','gold'
nc = 16                                     # 16, 32    for 1M or 4M parameters
S = 2                                       # Sampling steps
r = .8                                      # renoising time ratio
patch_size=3000                             # Maximum side of patches used if inference triggers memory error, to lower in case this happens.
char_size = 1024                            # character size
string='your_text'                          # use '_' to break line, all lines must have the same number of caracters, you can use blanks ' '
dilation = 2                                # iterations of morphological dilation on caracters mask

size=( (string.count('_')+1) * char_size , string.find('_')*char_size if string.find('_')!=-1 else len(string)*char_size)




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



# create mask 
width=21
sigma=5
kernel_size = [width,width] 
kernel = 1
mgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
kernel=(mgrids[0]-(kernel_size[0] - 1) / 2)**2+(mgrids[1]-(kernel_size[1] - 1) / 2)**2
kernel=torch.exp(-kernel/sigma**2)
kernel = kernel / torch.sum(kernel)
kernel = kernel.view(1, 1, *kernel.size())
conv = torch.nn.Conv2d(1,1, kernel_size, groups=1, bias=False, stride=1, padding=int((kernel_size[0] - 1) / 2), padding_mode='replicate')
conv.weight.data = kernel
conv.weight.requires_grad = False
conv.cuda()
kernel = np.ones((3,3),np.uint8)
mask=None
mask_list=[]
for i,s in enumerate(string.upper()):

    if s=='_':
        if mask is None:
            mask=torch.cat(mask_list,dim=-1)
        else:
            mask=torch.cat((mask,torch.cat(mask_list,dim=-1)),dim=-2)
        mask_list=[]

    elif s==' ':
        mask_letter=0.*transforms.ToTensor()(cv2.dilate(np.array(Image.open('./images/letters/1.png')),kernel,iterations = dilation)).unsqueeze(0)
        mask_list.append(mask_letter)
    else:
        mask_letter=transforms.ToTensor()(cv2.dilate(np.array(Image.open('./images/letters/%s.png'%s)),kernel,iterations = dilation)).unsqueeze(0)
        mask_list.append(mask_letter)
if mask is None:
    mask=torch.cat(mask_list,dim=-1)
else:
    try:
        mask=torch.cat((mask,torch.cat(mask_list,dim=-1)),dim=-2)
    except:
        pass


omega=mask.cuda()
omega=conv(omega).cpu()
save_image(omega[0],'./images/results/%s_mask.jpg'%string)

im=diffusion1.spatial_interp(size=size,time_ratio=r,omega=omega,patch_size=patch_size)
clear_output()
plt.figure(figsize=(15,15))
plt.imshow(im[0].permute(1,2,0).cpu())
plt.show()
save_image(im, './images/results/%s_%s_%s_32.jpg'%(string,name1,name2))

        
