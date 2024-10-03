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

## brain tissue segmentataion``
modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet_5fold_Adam_0.001/BestLoss_Fold2_0.009613921074196697.pth'
modelPath2 = './checkpoints/brain tissue/5fold_50epoch/UNet2+_5fold_Adam_0.001/BestLoss_Fold3_0.00946835994720459.pth'
modelPath3 = './checkpoints/brain tissue/5fold_50epoch/UNet3+_5fold_Adam_0.001/BestLoss_Fold4_0.009212232362478971.pth'
# modelPath = './checkpoints/brain tissue/5fold_50epoch/UNet2022_5fold_Adam_0.001/BestLoss_Fold2_0.009737449120730162.pth' 
input_path = '../dataset/brain_ISLES/test_ISLES/imgs_hard/'
# input_path = '../dataset/brain_ISLES/test_ISLES/imgs/'

## coco cat dataset
# modelPath = './checkpoints/cocodataset_cat/5fold_50epoch/UNet_5fold_Adam_0.001/BestLoss_Fold1_0.14052904068304786.pth'
# modelPath2 = './checkpoints/cocodataset_cat/5fold_50epoch/UNet2+_5fold_Adam_0.001/BestLoss_Fold4_0.14956443420935495.pth'
# modelPath3 = './checkpoints/cocodataset_cat/5fold_50epoch/UNet3+_5fold_Adagrad_0.001/BestLoss_Fold3_0.11211895272656147.pth'
# input_path = '../dataset/cocodataset_cat/testing/imgs/'


output_path = './output/'

# =============================================================================
# 參數設定
# =============================================================================
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# output_size = 512 # modified
output_size = 256 # modified

def fusion(img1, img2, img3, device):
	# Mean
	imgFusion = (torch.sigmoid(img1.to(device)) + torch.sigmoid(img2.to(device)) + torch.sigmoid(img3.to(device)))/3

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0
	
	img3 = torch.sigmoid(img3)
	img3[img3>0.5]=1
	img3[img3<=0.5]=0

	filt2 = img1.to(device).clone().detach().type(torch.int8) & img2.to(device).clone().detach().type(torch.int8) & img3.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion

def union(img1, img2, img3, device):
	imgFusion = (torch.sigmoid(img1.to(device)) + torch.sigmoid(img2.to(device)) + torch.sigmoid(img3.to(device)))/3

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0
	
	img3 = torch.sigmoid(img3)
	img3[img3>0.5]=1
	img3[img3<=0.5]=0

	filt2 = img1.to(device).clone().detach().type(torch.int8) | img2.to(device).clone().detach().type(torch.int8) | img3.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion

# 2024.09.11 add eigen value weight
def weighted_PCA(img1, img2, img3, device):
	# 判斷img tensor大小
	if len(img1.size()) == 4:
		img1 = img1.squeeze(0)
		img2 = img2.squeeze(0)
		img3 = img3.squeeze(0)

	if len(img1.size()) == 3:  # (C, H, W)
		img1_2d = img1.view(img1.size(1), img1.size(2)) # 只使用寬和高進行Covariance計算
		img2_2d = img2.view(img2.size(1), img2.size(2))
		img3_2d = img3.view(img3.size(1), img3.size(2))
	elif len(img1.size()) == 2:
		img1_2d = img1
		img2_2d = img2
		img3_2d = img3
	else:
		raise RuntimeError("Unexpected image dimensions: should be 2D or 3D.")

	img1_centered = img1_2d - torch.mean(img1_2d)  # 中心化
	img2_centered = img2_2d - torch.mean(img2_2d)  
	img3_centered = img3_2d - torch.mean(img3_2d)  
	
	# 矩陣相乘
	cov_matrix1 = torch.mm(img1_centered, img1_centered.t()) / (img1_2d.size(1) - 1)
	cov_matrix2 = torch.mm(img2_centered, img2_centered.t()) / (img2_2d.size(1) - 1)
	cov_matrix3 = torch.mm(img3_centered, img3_centered.t()) / (img3_2d.size(1) - 1)

    # 取出他的特徵值(eigenvalue)、特徵向量(eigenvector)
	eigenvalues1, eigenvectors1 = torch.linalg.eig(cov_matrix1)
	eigenvalues2, eigenvectors2 = torch.linalg.eig(cov_matrix2)
	eigenvalues3, eigenvectors3 = torch.linalg.eig(cov_matrix3)

	# 因為PCA特徵值通常最大都會占80-90%，所以這裡只提取最大特徵值及其對應的特徵向量
	max_eigenvalue1, max_index1 = torch.max(eigenvalues1.real, 0)
	max_eigenvalue2, max_index2 = torch.max(eigenvalues2.real, 0)
	max_eigenvalue3, max_index3 = torch.max(eigenvalues3.real, 0)

	if max_eigenvalue1 + max_eigenvalue2 + max_eigenvalue3 == 0:
		omega1 = omega2 = omega3 = 0.5
	else:
		omega1 = max_eigenvalue1 / (max_eigenvalue1 + max_eigenvalue2 + max_eigenvalue3)
		omega2 = max_eigenvalue2 / (max_eigenvalue1 + max_eigenvalue2 + max_eigenvalue3)
		omega3 = max_eigenvalue3 / (max_eigenvalue1 + max_eigenvalue2 + max_eigenvalue3)

	# return omega1, omega2, eigenvectors1[:, max_index1], eigenvectors2[:, max_index2]



	print("omega1 =", omega1, "omega2 =", omega2, "omega3 =", omega3)
	imgFusion = ( (torch.sigmoid(img1.to(device))*omega1) + (torch.sigmoid(img2.to(device))*omega2) + (torch.sigmoid(img3.to(device))*omega3) ) / (omega1 + omega2 + omega3)

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0

	img3 = torch.sigmoid(img3)
	img3[img3>0.5]=1
	img3[img3<=0.5]=0

	# intersection
	# filt2 = img1.to(device).clone().detach().type(torch.int8) & img2.to(device).clone().detach().type(torch.int8) & img3.to(device).clone().detach().type(torch.int8)
	# union
	filt2 = img1.to(device).clone().detach().type(torch.int8) | img2.to(device).clone().detach().type(torch.int8) | img3.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion

def predict_img(net,
				net2,
				full_img,
				# cbf,
				# cbv,
				# mtt,
				# tmax,
				device,
				scale_factor=1,
				out_threshold=0.5,
				folder='',
				flag='.png'):
	
	net.eval()
	net2.eval()
	if flag == '.png':
		img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
	else:
		full_img = (full_img - np.min(full_img)) / (np.max(full_img) - np.min(full_img))
		img = torch.as_tensor(full_img).unsqueeze(0)

		# cbf = (cbf - np.min(cbf)) / (np.max(cbf) - np.min(cbf))
		# cbf2 = torch.as_tensor(cbf).unsqueeze(0)

		# cbv = (cbv - np.min(cbv)) / (np.max(cbv) - np.min(cbv))
		# cbv2 = torch.as_tensor(cbv).unsqueeze(0)

		# mtt = (mtt - np.min(mtt)) / (np.max(mtt) - np.min(mtt))
		# mtt2 = torch.as_tensor(mtt).unsqueeze(0)

		# tmax = (tmax - np.min(tmax)) / (np.max(tmax) - np.min(tmax))
		# tmax2 = torch.as_tensor(tmax).unsqueeze(0)
	
	img = img.unsqueeze(0)

	# cbf2 = cbf2.unsqueeze(0)
	# cbv2 = cbv2.unsqueeze(0)
	# mtt2 = mtt2.unsqueeze(0)
	# tmax2 = tmax2.unsqueeze(0)
	# img = torch.cat((img, cbf2, cbv2, mtt2, tmax2), 1)

	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		output = net(img).cpu()
		output2 = net2(img).cpu()
		output3 = net3(img).cpu()

		## 2024.05.14 modified, 將回傳圖片更改為交集/聯集後的結果
		new = weighted_PCA(output, output2, output3, device)
		# new = fusion(output, output2, output3, device)
		# new = union(output, output2, output3, device)

		## 先放大成原圖的size再threshold
		if flag == '.png':
			# output = F.interpolate(output, (output_size, output_size), mode='bilinear') # 不使用output
			new = F.interpolate(output, (output_size, output_size), mode='bilinear') # 使用兩者交互作用的圖
		
		# mask = torch.sigmoid(output) > out_threshold # 不使用output
		mask = torch.sigmoid(new) > out_threshold # 使用兩者交互作用的圖

	return mask[0].long().squeeze().cpu().numpy() # 2024.05.14 modified, add Tensor.cpu()


def mask_to_image(mask):
	return Image.fromarray((mask.astype(np.uint8)*255).astype(np.uint8))


def get_args():
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--model', '-m', default=modelPath, metavar='FILE',
						help='Specify the files in which the model is stored')
	parser.add_argument('--model2', '-m2', default=modelPath2, metavar='FILE',
						help='Specify the files in which the model is stored')
	parser.add_argument('--model3', '-m3', default=modelPath3, metavar='FILE',
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
		out_files[i] = spl[0] + '.jpg' # 改副檔名
		# out_files[i] = spl[0] + '.png' # 改副檔名
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = torch.load(args.model, map_location=device)
	net = net.cuda(device)
	net.eval()

	net2 = torch.load(args.model2, map_location=device)
	net2 = net2.cuda(device)
	net2.eval()

	net3 = torch.load(args.model3, map_location=device)
	net3 = net3.cuda(device)
	net3.eval()

	logging.info("Model Load !")

	for i, fileName in enumerate(in_files):
		flag = splitext(fileName)[1] 
		if splitext(fileName)[1] == '.tiff':
			img = cv2.imread(input_path + str(fileName), cv2.IMREAD_UNCHANGED)

			# cbf = cv2.imread(dir_CBF + str(fileName), cv2.IMREAD_UNCHANGED)
			# cbv = cv2.imread(dir_CBV + str(fileName), cv2.IMREAD_UNCHANGED)
			# mtt = cv2.imread(dir_MTT + str(fileName), cv2.IMREAD_UNCHANGED)
			# tmax = cv2.imread(dir_Tmax + str(fileName), cv2.IMREAD_UNCHANGED)
		else:
			img = Image.open(input_path + fileName)
			img = img.convert('L')
			lx = transforms.Compose([
					transforms.Resize([output_size, output_size])
				])
			img = lx(img)

		# logging.info(out_files[i])
		mask = predict_img(net=net,
							net2=net2,
							full_img=img,
							# cbf=cbf,
							# cbv=cbv,
							# mtt=mtt,
							# tmax=tmax,
							scale_factor=args.scale,
							out_threshold=args.mask_threshold,
							device=device,
							flag=flag)

		if not args.no_save:
			out_filename = out_files[i]
			result = mask_to_image(mask)
			result.save(output_path+out_filename)
			logging.info('Mask saved to {}'.format(output_path+out_filename))
