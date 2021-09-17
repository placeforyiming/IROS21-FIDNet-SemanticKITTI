
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from network.ResNet import *
import argparse
from dataloader.Dataset_semanticKITTI import *
from utils import *
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP





parser = argparse.ArgumentParser()
#parameters for dataset
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="128")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--code_mode', dest= "code_mode", default="train", help="train or val or trainval")


# data_loader
parser.add_argument('--if_aug', dest= "if_aug", default=True, help="if if_aug")
parser.add_argument('--if_range_mask', dest= "if_range_mask", default=True, help="if if_range_mask")


# network settings
parser.add_argument('--backbone', dest= "backbone", default="ResNet34_point", help="ResNet34_aspp_1,ResNet34_aspp_2,ResNet_34_point")
parser.add_argument('--batch_size', dest= "batch_size", default=2, help="bs")
parser.add_argument('--if_BN', dest= "if_BN", default=True, help="if use BN in the backbone net")
parser.add_argument('--if_remission', dest= "if_remission", default=True, help="if concatenate remmision in the input")
parser.add_argument('--if_range', dest= "if_range", default=True, help="if concatenate range in the input")
parser.add_argument('--with_normal', dest= "with_normal", default=True, help="if concatenate normal in the input")



# training settins
parser.add_argument('--start_epoch',  dest= "start_epoch", default=0,help="0 or from the beginning, or from the middle")
parser.add_argument('--lr_policy',  dest= "lr_policy", default=1,help="lr_policy: 1, 2")
parser.add_argument('--total_epoch',  dest= "total_epoch", default=26,help="total_epoch")
parser.add_argument('--weight_WCE',  dest= "weight_WCE", default=1.0,help="weight_WCE")
parser.add_argument('--weight_LS',  dest= "weight_LS", default=3.0,help="weight_LS")
parser.add_argument('--top_k_percent_pixels',  dest= "top_k_percent_pixels", default=0.15,help="top_k_percent_pixels, hard mining")
parser.add_argument('--BN_train',  dest= "BN_train", default=True,help="if BN_train, false when batch_size is small")
parser.add_argument('--if_mixture',  dest= "if_mixture", default=True,help="if_mixture training")

# training mode

parser.add_argument('--if_multi_gpus',	dest= "if_multi_gpus", default=False,help="if_multi-gpus training")
parser.add_argument('--local_rank', default=-1,type=int)

args = parser.parse_args()

if args.if_multi_gpus:
	torch.cuda.set_device(args.local_rank)
	dist.init_process_group(backend='nccl')
	device=torch.device("cuda",args.local_rank)
else:
	device = torch.device('cuda:{}'.format(0))
	
save_path="./save_semantic/"
if not(os.path.exists(save_path)):
	os.mkdir(save_path)

temp_path=args.backbone+"_"+str(args.range_x)+"_"+str(args.range_y)+"_BN"+str(args.if_BN)+"_remission"+str(args.if_remission)+"_range"+str(args.if_range)+"_normal"+str(args.with_normal)+"_rangemask"+str(args.if_range_mask)+"_"+str(args.batch_size)+"_"+str(args.weight_WCE)+"_"+str(args.weight_LS)+"_lr"+str(args.lr_policy)+"_top_k"+str(args.top_k_percent_pixels)

save_path=save_path+temp_path+"/"
if not(os.path.exists(save_path)):
	os.mkdir(save_path)



dataset_train=Dataset_semanticKITTI(root=args.root,split=args.code_mode,is_train=True, range_img_size=(args.range_y,args.range_x),if_aug=args.if_aug, if_range_mask=args.if_range_mask,if_remission=args.if_remission, if_range=args.if_range, with_normal=args.with_normal)


if args.if_multi_gpus:
	train_sampler=torch.utils.data.distributed.DistributedSampler(dataset=dataset_train,drop_last=True)
	shuffle=False
else:
	train_sampler=RandomSampler(dataset_train)
	shuffle=False


data_loader_train = torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size,sampler=train_sampler,num_workers=2,shuffle=shuffle,pin_memory=True)

print(len(data_loader_train))




if args.backbone=="ResNet34_aspp_1":
	Backend=resnet34_aspp_1(if_BN=args.if_BN,if_remission=args.if_remission,if_range=args.if_range)
	S_H=SemanticHead(20,1152)

if args.backbone=="ResNet34_aspp_2":
	Backend=resnet34_aspp_2(if_BN=args.if_BN,if_remission=args.if_remission,if_range=args.if_range)
	S_H=SemanticHead(20,128*13)


if args.backbone=="ResNet34_point":
	Backend=resnet34_point(if_BN=args.if_BN,if_remission=args.if_remission,if_range=args.if_range,with_normal=args.with_normal)
	S_H=SemanticHead(20,1024)

model=Final_Model(Backend,S_H)

#if args.start_epoch>0:
#        model.load_state_dict(torch.load(save_path+str(args.start_epoch-1)))

if args.if_multi_gpus:
	model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model=DDP(model.to(device),device_ids=[args.local_rank],output_device=args.local_rank)
else:
	model.to(device)


if args.start_epoch>0:
	if args.if_multi_gpus:
		model.module.load_state_dict(torch.load(save_path+str(args.start_epoch-1)))
	else:
		model.load_state_dict(torch.load(save_path+str(args.start_epoch-1)))
	

weight=CE_Weight.get_weight()
weight=torch.tensor(weight).to(device)


WCE = nn.CrossEntropyLoss(weight=weight,ignore_index=255,reduction='none').to(device)
LS = Lovasz_softmax(ignore=0).to(device)

if args.BN_train:
	model.train()
else:
	model.eval()



optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.005)



scaler = torch.cuda.amp.GradScaler()

for current_epoch in range(args.start_epoch,args.total_epoch):
	if args.if_multi_gpus:
		data_loader_train.sampler.set_epoch(current_epoch)
	loss_per_epoch=0.0
	if args.lr_policy==1:
		learning_rate=get_lr_manually_1(current_epoch)
	if args.lr_policy==2:
		learning_rate=get_lr_manually_2(current_epoch)

	optimizer.param_groups[0]['lr']=learning_rate
	print ("lr=")
	print (learning_rate)

	print (save_path)

	for batch_ndx, (input_tensor,semantic_label,semantic_label_mask) in enumerate(data_loader_train):


		#print (np.shape(semantic_label))
		input_tensor=input_tensor.to(device)
		semantic_label=torch.squeeze(semantic_label,axis=1).to(device)
		semantic_label_mask=torch.squeeze(semantic_label_mask,axis=1)

		with torch.cuda.amp.autocast(enabled=args.if_mixture):
			semantic_output=model(input_tensor)

			pixel_losses = WCE(semantic_output, semantic_label) * semantic_label_mask.to(device)
			pixel_losses = pixel_losses.contiguous().view(-1)
			if args.top_k_percent_pixels == 1.0:
			    loss_ce=pixel_losses.mean()
			else:
			    top_k_pixels = int(args.top_k_percent_pixels * pixel_losses.numel())
			    pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
			    loss_ce=pixel_losses.mean()

			LS_loss=LS(TF.softmax(semantic_output, dim=1), semantic_label)
			total_loss=args.weight_WCE*loss_ce + args.weight_LS*LS_loss.mean()
		loss_per_epoch+=total_loss.item()

		optimizer.zero_grad()
		scaler.scale(total_loss).backward()
		#torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
		scaler.step(optimizer)
		scaler.update()
		input_tensor=None
		semantic_label=None
		semantic_label_mask=None
		semantic_output=None
		#total_loss.backward()
		#optimizer.step()
		
		if batch_ndx%100==0 and batch_ndx>0:
			print (loss_per_epoch/batch_ndx)	
		if batch_ndx%1000==0 and batch_ndx>0:
			print ("average loss for epoch "+str(current_epoch))
			print (loss_per_epoch/batch_ndx)

	if args.if_multi_gpus:
		if dist.get_rank()==0:
			torch.save(model.module.state_dict(),save_path+str(current_epoch))
	else:
		torch.save(model.state_dict(), save_path+str(current_epoch))

	

