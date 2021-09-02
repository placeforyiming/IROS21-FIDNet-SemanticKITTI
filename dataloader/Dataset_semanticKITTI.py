import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import random
import matplotlib.pyplot as plt
import yaml
import pickle
from .laserscan import SemLaserScan,LaserScan
import cv2
#rom laserscan import SemLaserScan




class Dataset_semanticKITTI(data.Dataset):
    

    def __init__(self,
                 root="./Dataset/semanticKITTI/",
                 split="train",
                 is_train=True,
                 range_img_size=(128, 2048),
                 if_aug=True,
                 if_range_mask=True,
                 if_remission=True,
                 if_range=True,
                 flip_sign=True,
                 with_normal=True

                 ):

        # root= ./Dataset/semanticKITTI/
        # split= 'train' or 'val' or 'trainval'
        self.root = root
        self.split = split
        self.is_train = is_train

        self.range_h, self.range_w = range_img_size

        self.if_aug = if_aug
        self.if_range_mask=if_range_mask
        self.if_remission=if_remission
        self.if_range=if_range
        self.flip_sign=flip_sign
        self.with_normal=with_normal
        

        self.CFG = yaml.safe_load(open(root+'semantic-kitti.yaml', 'r'))
        
        self.color_dict = self.CFG["color_map"]

        self.label_transfer_dict =self.CFG["learning_map"]

        self.nclasses = len(self.color_dict)

        self.A=SemLaserScan(nclasses=self.nclasses , sem_color_dict=self.color_dict, project=True, flip_sign=self.flip_sign, H=self.range_h, W=self.range_w, fov_up=3.0, fov_down=-25.0)


        if self.split=='train' or self.split=='val': 
            self.lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+self.split+'/*/*/*.bin')
        if self.split=='trainval':
            self.lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+'train'+'/*/*/*.bin')+glob.glob(root+'/data_odometry_velodyne/*/*/'+'val'+'/*/*/*.bin')

        self.label_list = [i.replace("velodyne", "labels") for i in self.lidar_list]

        self.label_list = [i.replace("bin", "label") for i in self.label_list]

        print (len(self.label_list))

        if self.with_normal:
            fov_up=3.0
            fov_down=-25.0
            self.fov_up=fov_up
            self.fov_down=fov_down
            fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
            fov_down = fov_down  / 180.0 * np.pi  # field of view down in rad
            fov = abs(fov_down) + abs(fov_up) 
            zero_matrix=np.zeros((self.range_h,self.range_w))
            one_matrix=np.ones((self.range_h,self.range_w))

            self.theta_channel=np.zeros((self.range_h,self.range_w))
            self.phi_channel=np.zeros((self.range_h,self.range_w))
            for i in range(self.range_h):
                for j in range(self.range_w):
                    self.theta_channel[i,j]=np.pi*(float(j+0.5)/self.range_w*2-1)
                    self.phi_channel[i,j]=(1-float(i+0.5)/self.range_h)*fov -abs(fov_down)
            self.R_theta=[np.cos(self.theta_channel),-np.sin(self.theta_channel),zero_matrix,np.sin(self.theta_channel),np.cos(self.theta_channel),zero_matrix,zero_matrix,zero_matrix,one_matrix]
            self.R_theta=np.asarray(self.R_theta)
            self.R_theta=np.transpose(self.R_theta, (1, 2, 0))
            self.R_theta=np.reshape(self.R_theta,[self.range_h,self.range_w,3,3])
            self.R_phi=[np.cos(self.phi_channel),zero_matrix,-np.sin(self.phi_channel),zero_matrix,one_matrix,zero_matrix,np.sin(self.phi_channel),zero_matrix,np.cos(self.phi_channel)]
            self.R_phi=np.asarray(self.R_phi)
            self.R_phi=np.transpose(self.R_phi, (1, 2, 0))
            self.R_phi=np.reshape(self.R_phi,[self.range_h,self.range_w,3,3])
            self.R_theta_phi=np.matmul(self.R_theta,self.R_phi)





    def __len__(self):
        return len(self.lidar_list)

    def __getitem__(self, index):
       
        self.A.open_scan(self.lidar_list[index])
        self.A.open_label(self.label_list[index])

        dataset_dict = {}

        dataset_dict['xyz'] = self.A.proj_xyz
        dataset_dict['remission'] = self.A.proj_remission
        dataset_dict['range_img'] = self.A.proj_range
    

        if self.if_range_mask:
            range_mask=(self.A.proj_range)/100.0
            dataset_dict['xyz_mask'] = self.A.proj_mask*range_mask
            range_mask=None
        else:
            dataset_dict['xyz_mask'] = self.A.proj_mask
        #dataset_dict['remission'] = F.to_tensor(self.A.proj_mask*(self.A.proj_remission-self.remission_mean)/self.remission_std)
        
        semantic_label= self.A.proj_sem_label
        instance_label= self.A.proj_inst_label
        x_y_z_img=self.A.proj_xyz
        
        semantic_train_label=self.generate_label(semantic_label)
        if self.with_normal:
            normal_image=self.calculate_normal(self.fill_spherical(self.A.proj_range))
            normal_image=normal_image*np.transpose([self.A.proj_mask,self.A.proj_mask,self.A.proj_mask],[1,2,0])
            dataset_dict['normal_image']=normal_image

        dataset_dict['semantic_label']=semantic_train_label
        if self.if_aug:
            split_point=random.randint(100,self.range_w-100)
            dataset_dict=self.sample_transform(dataset_dict,split_point)
        rand_mask_single=None
        rand_mask_multi=None


           

        input_tensor,semantic_label,semantic_label_mask=self.prepare_input_label_semantic(dataset_dict)


        #sample = {'input_tensor': input_tensor, 'semantic_label': semantic_label,'semantic_label_mask':semantic_label_mask}

        return  F.to_tensor(input_tensor), F.to_tensor(semantic_label).to(dtype=torch.long), F.to_tensor(semantic_label_mask)


    def prepare_input_label_semantic(self,sample):

        scale_x=np.expand_dims(np.ones([self.range_h, self.range_w])*50.0,axis=-1).astype(np.float32)
        scale_y=np.expand_dims(np.ones([self.range_h, self.range_w])*50.0,axis=-1).astype(np.float32)
        scale_z=np.expand_dims(np.ones([self.range_h, self.range_w])*3.0,axis=-1).astype(np.float32)
        scale_matrx=np.concatenate([scale_x,scale_y,scale_z],axis=2)
        if self.if_remission and not self.if_range:
            each_input=[sample['xyz']/scale_matrx,np.expand_dims(sample['remission'],axis=-1)]
            input_tensor=np.concatenate(each_input,axis=-1)
        if self.if_remission and self.if_range:
            each_input=[sample['xyz']/scale_matrx,np.expand_dims(sample['remission'],axis=-1),np.expand_dims(sample['range_img']/80.0,axis=-1)]
            input_tensor=np.concatenate(each_input,axis=-1)
        if not self.if_remission and not self.if_range:
            input_tensor=sample['xyz']/scale_matrx
        semantic_label=sample['semantic_label'][:,:]
        semantic_label_mask=sample['xyz_mask'][:,:]
        if self.with_normal:
            input_tensor=np.concatenate([input_tensor,sample['normal_image'].astype(np.float32)],axis=-1)

        



        


        #plt.imsave('./hhh.jpg',(normal_image+1.0)/2.0)

        '''
        random_mask=np.random.randint(20, size=(self.range_h, self.range_w))
        random_mask=random_mask<18
        semantic_label=semantic_label*random_mask
        _,_,n_channel=np.shape(input_tensor)
        random_mask=np.expand_dims(random_mask,axis=-1)
        random_mask=np.tile(random_mask,[1,1,n_channel])
        input_tensor=input_tensor*random_mask
        '''
        return input_tensor,semantic_label,semantic_label_mask

    def sample_transform(self,dataset_dict,split_point):
        dataset_dict['xyz']=np.concatenate([dataset_dict['xyz'][:,split_point:,:],dataset_dict['xyz'][:,:split_point,:]],axis=1)

        dataset_dict['xyz_mask']=np.concatenate([dataset_dict['xyz_mask'][:,split_point:],dataset_dict['xyz_mask'][:,:split_point]],axis=1)

        dataset_dict['remission']=np.concatenate([dataset_dict['remission'][:,split_point:],dataset_dict['remission'][:,:split_point]],axis=1)
        dataset_dict['range_img']=np.concatenate([dataset_dict['range_img'][:,split_point:],dataset_dict['range_img'][:,:split_point]],axis=1)

        dataset_dict['semantic_label']=np.concatenate([dataset_dict['semantic_label'][:,split_point:],dataset_dict['semantic_label'][:,:split_point]],axis=1)
        if self.with_normal:
            dataset_dict['normal_image']=np.concatenate([dataset_dict['normal_image'][:,split_point:,:],dataset_dict['normal_image'][:,:split_point,:]],axis=1)
        

        return dataset_dict






    def sem_label_transform(self,raw_label_map):
        for i in self.label_transfer_dict.keys():
            raw_label_map[raw_label_map==i]=self.label_transfer_dict[i]
        return raw_label_map

    def generate_label(self,semantic_label):

        original_label=np.copy(semantic_label)
        label_new=self.sem_label_transform(original_label)
        
        return label_new


    def fill_spherical(self,range_image):
        # fill in spherical image for calculating normal vector
        height,width=np.shape(range_image)[:2]
        value_mask=np.asarray(1.0-np.squeeze(range_image)>0.1).astype(np.uint8)
        dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)

        with_value=np.squeeze(range_image)>0.1
            
        depth_list=np.squeeze(range_image)[with_value]
            
        label_list=np.reshape(lbl,[1,height*width])
        depth_list_all=depth_list[label_list-1]

        depth_map=np.reshape(depth_list_all,(height,width))
        
        depth_map = cv2.GaussianBlur(depth_map,(7,7),0)
        depth_map=range_image*with_value+depth_map*(1-with_value)
        return depth_map

    def calculate_normal(self,range_image):
        
        one_matrix=np.ones((self.range_h,self.range_w))
        #img_gaussian =cv2.GaussianBlur(range_image,(3,3),0)
        img_gaussian =range_image
        #prewitt
        kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        self.partial_r_theta=img_prewitty /(np.pi*2.0/self.range_w)/6
        self.partial_r_phi=img_prewittx/(((self.fov_up-self.fov_down)/180.0*np.pi)/self.range_h)/6



        partial_vector=[1.0*one_matrix,self.partial_r_theta/(range_image*np.cos(self.phi_channel)),self.partial_r_phi/range_image]
        partial_vector=np.asarray(partial_vector)
        partial_vector=np.transpose(partial_vector, (1, 2, 0))
        partial_vector=np.reshape(partial_vector,[self.range_h,self.range_w,3,1])
        normal_vector=np.matmul(self.R_theta_phi,partial_vector)
        normal_vector=np.squeeze(normal_vector)
        normal_vector=normal_vector/np.reshape(np.max(np.abs(normal_vector),axis=2),(self.range_h,self.range_w,1))
        normal_vector_camera=np.zeros((self.range_h,self.range_w,3))
        normal_vector_camera[:,:,0]=normal_vector[:,:,1]
        normal_vector_camera[:,:,1]=-normal_vector[:,:,2]
        normal_vector_camera[:,:,2]=normal_vector[:,:,0]
        return normal_vector_camera
                    

