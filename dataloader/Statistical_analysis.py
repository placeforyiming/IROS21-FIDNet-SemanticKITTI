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





root= "../Dataset/semanticKITTI/"
split= 'train'

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

	A.open_scan(lidar_list[i])
	A.open_label(label_list[i])

	original_label=np.copy(A.proj_sem_label)
	label_new=sem_label_transform(original_label)
	original_label=None

	center_list=[]
	individual_points=[]
	individual_colors=[]
	all_inst=np.unique(A.proj_inst_label)
	for j in all_inst:
		instance_mask_=(A.proj_inst_label==j)
		if j==0 :
			continue
		else:
			for mmm in np.unique(A.proj_sem_label):
				
				instance_mask=np.logical_and(np.logical_and(A.proj_sem_label==mmm,label_new>-1),instance_mask_)

				if np.sum(instance_mask)<30:
					continue

				temp_points=[]
				temp_colors=[]
				yyy,xxx=np.where(instance_mask>0)
				for kkk in range(len(xxx)):
					temp_colors.append([A.proj_sem_color[yyy[kkk],xxx[kkk],0],A.proj_sem_color[yyy[kkk],xxx[kkk],1],A.proj_sem_color[yyy[kkk],xxx[kkk],2]])
					temp_points.append([A.proj_xyz[yyy[kkk],xxx[kkk],0],A.proj_xyz[yyy[kkk],xxx[kkk],1],A.proj_xyz[yyy[kkk],xxx[kkk],2]])
					
				individual_points.append(temp_points)
				individual_colors.append(temp_colors)
				#range_mask=(A.proj_range-np.mean(A.proj_range[center_mask]))<15
				#instance_mask=np.logical_and(instance_mask,range_mask)			

				temp_center_x=(np.max(A.proj_xyz[:,:,0][instance_mask])+np.min(A.proj_xyz[:,:,0][instance_mask]))/2.0

				temp_center_y=(np.max(A.proj_xyz[:,:,1][instance_mask])+np.min(A.proj_xyz[:,:,1][instance_mask]))/2.0

				temp_center_z=(np.max(A.proj_xyz[:,:,2][instance_mask])+np.min(A.proj_xyz[:,:,2][instance_mask]))/2.0

				center_list.append([temp_center_x,temp_center_y,temp_center_z])
				temp_max_x=np.max(np.abs(A.proj_xyz[:,:,0]-temp_center_x)*instance_mask)
				temp_max_y=np.max(np.abs(A.proj_xyz[:,:,1]-temp_center_y)*instance_mask)
				temp_max_z=np.max(np.abs(A.proj_xyz[:,:,2]-temp_center_z)*instance_mask)

				temp_dis=np.max(np.sqrt((np.abs(A.proj_xyz[:,:,0]-temp_center_x)*instance_mask)**2+(np.abs(A.proj_xyz[:,:,1]-temp_center_y)*instance_mask)**2+(np.abs(A.proj_xyz[:,:,2]-temp_center_z)*instance_mask)**2))


				Big_car_center_dis_list.append([temp_max_x,temp_max_y,temp_max_z])
				max_dis=max(max_dis,temp_dis)

				if temp_max_x>5 or temp_max_y>5:
					count+=1
					pcd = o3d.geometry.PointCloud()
					pcd.points = o3d.utility.Vector3dVector(temp_points)
					pcd.colors = o3d.utility.Vector3dVector(temp_colors)
					o3d.io.write_point_cloud("./KITTI_original.ply", pcd)
					if "09" in lidar_list[i].split("/"):
						count_error+=1
					outlier_label=np.unique(A.proj_sem_label[instance_mask])[0]

				else:

					max_x=max(max_x,temp_max_x)
					max_y=max(max_y,temp_max_y)
					max_z=max(max_z,temp_max_z)

	smallest_dis_each_frame=100.0

	for k in range(len(center_list)):
		for l in range(len(center_list)):

			temp_min_x=np.abs(center_list[k][0]-center_list[l][0])
			temp_min_y=np.abs(center_list[k][1]-center_list[l][1])
			temp_min_z=np.abs(center_list[k][2]-center_list[l][2])
			temp_distance=np.sqrt(temp_min_x**2+temp_min_y**2+temp_min_z**2)/3
			if temp_min_x>0 and temp_min_x<min_center_x:
				min_center_x=temp_min_x
			if temp_min_y>0 and temp_min_y<min_center_y:
				min_center_y=temp_min_y
			if temp_min_z>0 and temp_min_z<min_center_z:
				min_center_z=temp_min_z
			if temp_distance>0 and temp_distance<min_distance:
				min_distance=temp_distance
			if temp_distance>0 and temp_distance<smallest_dis_each_frame:
				smallest_dis_each_frame=temp_distance

			'''
			if temp_distance<0.5 and temp_distance>0.0:
				print (temp_distance)
				plt.imsave("./ss.png",A.proj_sem_color)
			'''
	Each_frame_center_dis_list.append(smallest_dis_each_frame)

	
	print ("max_x:")
	print (max_x)
	print ("max_y:")
	print (max_y)
	print ("max_z:")
	print (max_z)
	print ("min_center_x:")
	print (min_center_x)
	print ("min_center_y:")
	print (min_center_y)
	print ("min_center_z:")
	print (min_center_z)
	




	print ("min_distance:")
	print (min_distance)
	
	print ("!!!!!!!!!")
	print (count)
	print (count_error)
	print (i)
	print (outlier_label)
	print ("!!!")

'''
max_x:
3.9558477
max_y:
3.3740053
max_z:
1.6115439
min_center_x:
0.0007238388061523438
min_center_y:
3.1948089599609375e-05
min_center_z:
7.62939453125e-06
min_distance:
0.09379423095148133
'''