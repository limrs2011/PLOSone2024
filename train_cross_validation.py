import argparse
import logging
import torch
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.model_selection import KFold # modified, added

from utils.dataset import BasicDataset, BasicDataset2
from ranger2020 import Ranger
from model import UNet_3Plus, unet2022, TripleUnet
from matplotlib import pyplot as plt # modified, added
from datetime import datetime # modified, added

from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss, DiceCELoss, GeneralizedDiceLoss
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
import segmentation_models_pytorch as smp

# ===============================================================================================
#  路徑設定
# ===============================================================================================

## brain tissue segmentation
dir_train_img = '../dataset/brain_ISLES/train_ISLES/imgs/'
dir_train_mask = '../dataset/brain_ISLES/train_ISLES/masks/'

## coco cat dataset
# dir_train_img = '../dataset/cocodataset_cat/training/imgs/'
# dir_train_mask = '../dataset/cocodataset_cat/training/masks/'

dir_checkpoint = 'checkpoints/'

# ===============================================================================================
# 參數設定
# ===============================================================================================
epoch = 50 # modified
# epoch = 100
epoch_store = 100 # how many epoch to store
batch = 4
# lr = 0.0001
lr = 0.001 # modified
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=epoch, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=batch, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=lr,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()

def train_model(device, epochs=50, batch_size=4, learning_rate=0.001,
				img_scale=1, save_checkpoint=True, amp=False, gradient_clipping=1.0):
	
	# 2024.08.26 BasicDataset輸出的是dataset，更改變數名 train-> dataset
	dataset = BasicDataset(dir_train_img, dir_train_mask, img_scale, transform=True) ## CT: 1 channel
	# dataset = BasicDataset2(dir_train_img, dir_train_mask, img_scale, dir_CBF, dir_CBV, dir_MTT, dir_Tmax, transform=True) ## CT+CTP: 5 channels

	n_train = len(dataset)
	# loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False)    # 2024.08.26 marked,交叉驗證的話不要在這裡定義
	# train_loader = DataLoader(train, shuffle=True, **loader_args, drop_last=True) # 2024.08.26 marked,交叉驗證的話不要在這裡分

	logging.info(f'''Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Training rate:   {n_train}
		Checkpoints:     {save_checkpoint}
		Device:          {device}
		Images scaling:  {img_scale}
		Mixed Precision: {amp}
	''')

	# global_step = 0 # 沒用到, marked
	# best_dice = 10. # 沒用到, marked
	# best_model_params = copy.deepcopy(model.state_dict())
	loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False) # 2024.08.26 added
	kf = KFold(n_splits=5, shuffle=True, random_state=42) # KFold parameter, random seed=42

	# KFold cross-validation
	for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
		logging.info(f'----- FOLD {fold+1} -----')
		best_loss = 10000.
		training_curve = [] ## 2024.05.02 modified, add training_curve
		validation_curve = [] ## 2024.05.02 modified, add validation_curve

		# subset and dataloader
		train_subset = Subset(dataset, train_idx)
		val_subset = Subset(dataset, val_idx)
		train_loader = DataLoader(train_subset, shuffle=True, **loader_args, drop_last=True)
		val_loader = DataLoader(val_subset, shuffle=False, **loader_args, drop_last=False)
		
		# 2024.08.26 model要在裡面定義，不然cross validation沒有比較的意義
		## =================================================================================================
		## Model
		## =================================================================================================
		
		## model = smp.Unet(in_channels=1, classes=1)
		# model = smp.Unet(
		# 	encoder_name = "vgg16",
		# 	encoder_weights= None,
		# 	in_channels=1, 
		# 	classes=1,
		# 	decoder_use_batchnorm=False
		# )

		## model = UNet_2Plus(in_channels=1, n_classes=1) # UNet2+改成用smp
		# model = smp.UnetPlusPlus(
		# 	encoder_name = "vgg16",
		# 	encoder_weights= None,
		# 	in_channels=1, 
		# 	classes=1,
		# 	decoder_use_batchnorm=False
		# )

		# model = UNet_3Plus(in_channels=1, n_classes=1)

		model = unet2022(n_channels=1, n_classes=1)

		model = nn.DataParallel(model)
		model = model.cuda(device) # 沒顯卡記得mark

		## =================================================================================================
		## Optimizer!!!!!!!!!!
		## =================================================================================================
		# optimizer = Ranger(model.parameters(), lr=learning_rate) 
		# optimizer = torch.optim.Adagrad(params=model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=1e-8) # modified
		optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-8)
		
		## =================================================================================================
		## Scheduler!!!!!!!!!!
		## =================================================================================================
		# scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-5, total_iters=epochs)
		# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=3, verbose=True)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
		
		## =================================================================================================
		## Loss Function
		## =================================================================================================
		# criterion = DiceCELoss(sigmoid=True, lambda_dice=1, lambda_ce=1)
		criterion = nn.BCEWithLogitsLoss() # modified, BCE loss 



		# iteration start
		for epoch in range(1, epochs+1):
			model.train()
			with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
				epoch_loss = 0
				num_batch = 0
				
				for batch in train_loader:
					num_batch += 1
					images, true_masks = batch['image'], batch['mask']

					images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
					true_masks = true_masks.to(device=device, dtype=torch.float32)

					with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
						masks_pred = model(images)
						# print(masks_pred.shape)
						# print(masks_pred)
						loss = criterion(masks_pred, true_masks)

					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
					optimizer.step() ## 更新權重

					pbar.update(images.shape[0])
					# global_step += 1
					epoch_loss += loss.item()
					pbar.set_postfix(**{'loss (batch)': loss.item()})

				scheduler.step(epoch_loss)
				logging.info('Train Loss: {}'.format(epoch_loss/num_batch))
				training_curve.append(epoch_loss/num_batch) ## 2024.05.02 modified

				# 2024.08.26 沒有習慣用這個，所以marked
				# with open('./txt/train_loss.txt', 'a') as f:
				# 	f.write(str(epoch_loss/num_batch))
				# 	f.write('\n')
				# 	f.close()

				## store best checkpoint
				if save_checkpoint and (epoch_loss/num_batch) < best_loss:
					best_loss = epoch_loss/num_batch
					best_model_params = copy.deepcopy(model.state_dict())
					logging.info('Save Best Model: {}!!'.format(epoch_loss/num_batch))
				
				## 2024.08.26 modified, add model eval
				model.eval()
				val_loss = 0
				num_batch = 0
				with torch.no_grad():
					for batch in val_loader:
						num_batch += 1
						images, true_masks = batch['image'], batch['mask']
						images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
						true_masks = true_masks.to(device=device, dtype=torch.float32)
						masks_pred = model(images)
						val_loss += criterion(masks_pred, true_masks).item()
					validation_curve.append(val_loss/num_batch) ## 2024.08.26 modified, added

			## 每隔多少epoch_store存一次模型 # 2024.06.04 modified, 先把它marked
			# if save_checkpoint and (epoch % epoch_store == 0):
			# 	torch.save(model, dir_checkpoint + f'Epoch{epoch}_valDice_{epoch_loss/num_batch}.pth')
			# 	logging.info('Checkpoint {} save !'.format(epoch))

		model.load_state_dict(best_model_params)
		torch.save(model, dir_checkpoint + f'BestLoss_Fold{fold+1}_{best_loss}.pth')
		
		# 2024.05.02 modified, add matplot
		base_path = os.path.dirname(os.path.abspath(__file__))
		x = list(range(1, epochs+1))
		# y = training_curve
		plt.clf() # clear
		plt.plot(x,training_curve, x,validation_curve)
		plt.xlabel("Epoch")
		plt.ylabel("Training_Loss")
		plt.grid(True)
		plt.savefig(os.path.join(base_path, "plt",  datetime.today().strftime('%Y%m%d%H%M%S') + f"_plt_trainFold{fold+1}.png"))
		# 2024.05.02 modified, add matplot

if __name__ == '__main__':
	args = get_args()
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')


	# model = smp.Unet(
	# 	encoder_name = "vgg16",
	# 	encoder_weights= None,
	# 	in_channels=1, 
	# 	classes=1,
	# 	decoder_use_batchnorm=False
	# )
	# model = UNet_2Plus(in_channels=1, n_classes=1)
	# model = UNet_3Plus(in_channels=1, n_classes=1)
	# model = unet2022(n_channels=1, n_classes=1)

	# model = smp.DeepLabV3Plus(
	# 	encoder_name = "efficientnet-b3",
	# 	encoder_weights= None,
	# 	in_channels=1, 
	# 	classes=1
	# )

	# model = UNETR(in_channels=1, out_channels=1, img_size=256, feature_size=32, norm_name=("group", {"num_groups": 16}), spatial_dims=2)
	# model = SwinUNETR(in_channels=1, out_channels=1, img_size=256, feature_size=48, norm_name=("group", {"num_groups": 12}), spatial_dims=2)

	# model = TripleUnet(
	# 	encoder_name = "vgg16",
	# 	encoder_weights= None,
	# 	# 1=Ensemble, 2=Concat, 3=Ensemble Method
	# 	in_channels=1, 
	# 	classes=1, 
	# 	decoder_use_batchnorm=False
	# )

	# model = unet2022(n_channels=1, n_classes=1)

	# model = nn.DataParallel(model)
	# model = model.cuda(device) # 沒顯卡記得mark

	# 2024.05.21 modified, 新增module，讓他變單顆GPU，因為single model這樣跑比較快，不用正規化
	# ref: https://blog.csdn.net/qq_39624083/article/details/131614071
	# model = model.module.cuda(device) # DeepLab3plus專用

	# 2024.08.26 modified, 因為沒必要去儲存KeyboardInterrupt，所以try except marked
	# try:
	train_model(
		# model = model, # modified
		epochs = args.epochs,
		batch_size = args.batch_size,
		learning_rate = args.lr,
		device = device,
		img_scale = args.scale,
		amp = args.amp
	)
	# except KeyboardInterrupt:
	# 	torch.save(model, 'INTERUPT.pth')
	# 	try:
	# 		sys.exit(0)
	# 	except SystemExit:
	# 		os._exit(0)


