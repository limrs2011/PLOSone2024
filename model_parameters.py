import sys
import os
import numpy as np

import torch
from model import UNet_3Plus, unet2022, TripleUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
import segmentation_models_pytorch as smp

## model information
model = smp.Unet(
		encoder_name = "vgg16",
		encoder_weights= None,
		in_channels=1, 
		classes=1,
		decoder_use_batchnorm=False
	)
# model = smp.UnetPlusPlus(
# 			encoder_name = "vgg16",
# 			encoder_weights= None,
# 			in_channels=1, 
# 			classes=1,
# 			decoder_use_batchnorm=False
# 		)
# model = UNet_3Plus(in_channels=1, n_classes=1)
# model = unet2022(n_channels=1, n_classes=1)

## model path
modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adam_0.001/BestLoss_Fold2_0.009613921074196697.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adam_0.001/BestLoss_Fold3_0.00946835994720459.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adam_0.001/BestLoss_Fold4_0.009212232362478971.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2022_5fold_Adam_0.001/BestLoss_Fold2_0.009737449120730162.pth' 

# load .pth file
checkpoint = torch.load(modelPath)



if isinstance(checkpoint, torch.nn.DataParallel):
    checkpoint = checkpoint.module.state_dict()
elif 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

if 'module.' in list(checkpoint.keys())[0]:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint)

total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params/1000000}")