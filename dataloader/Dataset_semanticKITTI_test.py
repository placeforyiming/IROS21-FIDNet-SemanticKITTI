import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from Dataset_semanticKITTI import Dataset_semanticKITTI
from laserscan import SemLaserScan,LaserScan
#rom laserscan import SemLaserScan



A=Dataset_semanticKITTI(root="../Dataset/semanticKITTI/")
dict_test=A[1000]
print (np.unique(dict_test['x_label']))

print (np.unique(dict_test['x_label']*dict_test['instance_bin_mask']))

'''
data_loader_train = torch.utils.data.DataLoader(A,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,collate_fn=lambda x: x)

for batch_ndx, sample in enumerate(data_loader_train):
	print (batch_ndx)
	print (sample.keys())
'''