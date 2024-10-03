import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from utils.dataset import BasicDataset
from os.path import splitext, isfile, join
import cv2

# =============================================================================
#  路徑設定
# =============================================================================

#region brain tissue segmentation
## Unet
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adagrad_0.001/BestLoss_Fold1_0.009483013832941652.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adagrad_0.0001/BestLoss_Fold4_0.015353552112355829.pth'
modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adam_0.001/BestLoss_Fold2_0.009613921074196697.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adam_0.0001/BestLoss_Fold4_0.009404093874618411.pth'

## Unet2+
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adagrad_0.001/BestLoss_Fold1_0.009489778270944953.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adagrad_0.0001/BestLoss_Fold1_0.013240453023463487.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adam_0.001/BestLoss_Fold3_0.00946835994720459.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adam_0.0001/BestLoss_Fold2_0.00936326286289841.pth'

## Unet3+
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adagrad_0.001/BestLoss_Fold1_0.008121352735906839.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adagrad_0.0001/BestLoss_Fold1_0.010019294265657664.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adam_0.001/BestLoss_Fold4_0.009212232362478971.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adam_0.0001/BestLoss_Fold2_0.008590910453349351.pth'

## UNet2022
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2022_5fold_Adam_0.001/BestLoss_Fold2_0.009737449120730162.pth' 
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2022_5fold_Adam_0.0001/BestLoss_Fold2_0.01004977349191904.pth'

#endregion


#region testing datapath
## brain segmentation
input_path = '../dataset/brain_ISLES/test_ISLES/imgs_hard/'
# input_path = '../dataset/brain_ISLES/test_ISLES/imgs/'

## coco cat dataset
# modelPath = './checkpoints/cocodataset_cat/5fold_50epoch/UNet_5fold_Adam_0.001/BestLoss_Fold1_0.14052904068304786.pth'
# modelPath = './checkpoints/cocodataset_cat/5fold_50epoch/UNet2+_5fold_Adagrad_0.001/BestLoss_Fold2_0.10732171091883161.pth'
# modelPath = './checkpoints/cocodataset_cat/5fold_50epoch/UNet3+_5fold_Adagrad_0.001/BestLoss_Fold3_0.11211895272656147.pth'
# input_path = '../dataset/cocodataset_cat/testing/imgs/'

#endregion


output_path = './output/'

# =============================================================================
# 參數設定
# =============================================================================
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# output_size = 512 # modified
output_size = 256 # modified

def predict_img(net,
				full_img,
				device,
				scale_factor=1,
				out_threshold=0.5,
				folder='',
				flag='.png'):
	
	net.eval()
	if flag == '.png':
		img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
	else:
		full_img = (full_img - np.min(full_img)) / (np.max(full_img) - np.min(full_img))
		img = torch.as_tensor(full_img).unsqueeze(0)
	
	img = img.unsqueeze(0)
	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		output = net(img).cpu()

		## 先放大成原圖的size再threshold
		if flag == '.png':
			logging.info("png mode")
			output = F.interpolate(output, (output_size, output_size), mode='bilinear')
		mask = torch.sigmoid(output) > out_threshold

	return mask[0].long().squeeze().numpy()


def mask_to_image(mask):
	return Image.fromarray((mask.astype(np.uint8)*255).astype(np.uint8))


def get_args():
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--model', '-m', default=modelPath, metavar='FILE',
						help='Specify the files in which the model is stored')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
						help='Minimum probability value to consider a mask pixel white')
	parser.add_argument('--scale', '-s', type=float, default=1,
						help='Scale factor for the input images')
	parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
	parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
	return parser.parse_args()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	args = get_args()
	in_files = next(os.walk(input_path))[2]
	out_files = in_files.copy()

	for i in range(len(out_files)):
		spl = out_files[i].split('.', 1)
		# out_files[i] = spl[0] + '.jpg' # 改副檔名
		out_files[i] = spl[0] + '.png' # 改副檔名

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	net = torch.load(args.model, map_location=device)
	net = net.cuda(device)
	net.eval()
	logging.info("Model Load !")

	for i, fileName in enumerate(in_files):
		flag = splitext(fileName)[1] 
		if splitext(fileName)[1] == '.tiff':
			img = cv2.imread(input_path + str(fileName), cv2.IMREAD_UNCHANGED)
		else:
			img = Image.open(input_path + fileName)
			img = img.convert('L')
			lx = transforms.Compose([
					transforms.Resize([output_size, output_size])
				])
			img = lx(img)

		mask = predict_img(net=net,
							full_img=img,
							scale_factor=args.scale,
							out_threshold=args.mask_threshold,
							device=device,
							flag=flag)

		if not args.no_save:
			out_filename = out_files[i]
			result = mask_to_image(mask)
			result.save(output_path+out_filename)
			logging.info('Mask saved to {}'.format(output_path+out_filename))
