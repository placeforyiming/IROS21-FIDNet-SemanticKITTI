import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from laserscan import SemLaserScan,LaserScan
import yaml




root= "../Dataset/semanticKITTI/"
split= 'train'

lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+split+'/*/*/*.bin')
       

label_list = [i.replace("velodyne", "labels") for i in lidar_list]
label_list = [i.replace("bin", "label") for i in label_list]
       

CFG = yaml.safe_load(open('./semantic-kitti.yaml', 'r'))
        
color_dict = CFG["color_map"]

label_transfer_dict =CFG["learning_map"]

nclasses = len(color_dict)

def sem_label_transfor(raw_label_map):
    for i in label_transfer_dict.keys():
        raw_label_map[raw_label_map==i]=label_transfer_dict[i]
    return raw_label_map

A=SemLaserScan(nclasses=nclasses , sem_color_dict=color_dict, project=True, H=128, W=2048, fov_up=3.0, fov_down=-25.0)

A.open_scan(lidar_list[10001])
A.open_label(label_list[10001])


print (np.mean(np.logical_or(A.proj_sem_label>32,A.proj_sem_label<10)))

print (np.mean(A.proj_inst_label==0))

print (np.sum(A.proj_xyz[:,:,2]==-1))

label_new=sem_label_transfor(A.proj_sem_label)
print (np.unique(label_new))
#plt.imshow(A.proj_range)
#plt.show()