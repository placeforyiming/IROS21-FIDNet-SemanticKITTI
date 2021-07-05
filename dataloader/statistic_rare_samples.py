import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from laserscan import SemLaserScan,LaserScan
import yaml
import open3d as o3d


all_label={}
for i in range(19):
	all_label[i+1]=[]


max_add_count=1000


root= "../Dataset/semanticKITTI/"
split= 'val'

lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+split+'/*/*/*.bin')
       

label_list = [i.replace("velodyne", "labels") for i in lidar_list]
label_list = [i.replace("bin", "label") for i in label_list]
       
thing_list=[1,2,3,4,5,6,7,8]
CFG = yaml.safe_load(open('./semantic-kitti.yaml', 'r'))
        
color_dict = CFG["color_map"]

label_transfer_dict =CFG["learning_map"]

nclasses = len(color_dict)
def sem_label_transform(raw_label_map):
    for i in label_transfer_dict.keys():
    	pre_map=raw_label_map==i
    	raw_label_map[pre_map]=label_transfer_dict[i]
    return raw_label_map

Big_car_center_dis_list=[]

Each_frame_center_dis_list=[]


A=SemLaserScan(nclasses=nclasses , sem_color_dict=color_dict, project=True, H=128, W=2048, fov_up=3.0, fov_down=-25.0)


max_x=0.0
max_y=0.0
max_z=0.0
max_dis=0.0



min_center_x=100.0
min_center_y=100.0
min_center_z=100.0
min_distance=100.0
outlier_label=0
count=0
count_error=0
for i in range(len(lidar_list)):
	print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	print (i)
	A.open_scan(lidar_list[i])
	A.open_label(label_list[i])

	original_label=np.copy(A.proj_sem_label)
	label_new=sem_label_transform(original_label)
	original_label=None

	for j in all_label.keys():
		print (j)
		if np.sum(label_new==j)>0:
			all_label[j].append(np.sum(label_new==j))
		print (len(all_label[j]))

	print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")







'''
19129
1
17774
2     *2
3377
3     *3
2794
4     *4
2255
5     *2
5015
6     *2
4678
7     *6
1171
8     *12
550
9
19130
10    
7665
11
18051
12    *2
5189
13
17104
14
18739
15
19130
16
17086
17
18521
18
18609
19
13147

'''
