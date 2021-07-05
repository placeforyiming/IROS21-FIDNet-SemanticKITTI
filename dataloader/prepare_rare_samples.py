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
import pickle
import random
#19*2 12*2 8*6 7*4 6*2 5*2 3*2 2*2 




root= "../Dataset/semanticKITTI/"
split= 'train'

lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+split+'/*/*/*.bin')
       

label_list = [i.replace("velodyne", "labels") for i in lidar_list]
label_list = [i.replace("bin", "label") for i in label_list]
 
new_lidar_list=[]
new_label_list=[]

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


for i in range(len(lidar_list)):

	print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	print (i)
	A.open_scan(lidar_list[i])
	A.open_label(label_list[i])


	original_label=np.copy(A.proj_sem_label)
	label_new=sem_label_transform(original_label)
	original_label=None

	temp_lidar=lidar_list[i][1:]
	temp_label=label_list[i][1:]
	


	#new_lidar_list.append(lidar_list[i][1:])
	#new_label_list.append(label_list[i][1:])
	if np.sum(label_new==2)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		
	if np.sum(label_new==3)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		
	if np.sum(label_new==4)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])

		
	if np.sum(label_new==5)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		
	if np.sum(label_new==6)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])

	if np.sum(label_new==7)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		
	if np.sum(label_new==8)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
		
	if np.sum(label_new==12)>0:
		new_lidar_list.append(lidar_list[i][1:])
		new_label_list.append(label_list[i][1:])
			
random.shuffle(new_lidar_list)
random.shuffle(new_label_list)

'''
with open("./save_new_lidar_path.txt", "wb") as fp:   #Pickling
   pickle.dump(new_lidar_list, fp)
	
with open("./save_new_label_path.txt", "wb") as fp:   #Pickling
   pickle.dump(new_label_list, fp)
'''

with open("./save_new_lidar_path_extra_only.txt", "wb") as fp:   #Pickling
   pickle.dump(new_lidar_list, fp)
	
with open("./save_new_label_path_extra_only.txt", "wb") as fp:   #Pickling
   pickle.dump(new_label_list, fp)

with open("./save_new_lidar_path.txt", "rb") as fp:   #Pickling
   new_lidar_list=pickle.load(fp)
	
with open("./save_new_label_path.txt", "rb") as fp:   #Pickling
   new_label_list=pickle.load(fp)



print (len(new_lidar_list))

print (len(new_label_list))



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






