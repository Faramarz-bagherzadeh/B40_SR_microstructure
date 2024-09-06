import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
import numpy as np
from torchvision import transforms
import tifffile
from patchify import patchify, unpatchify
import kornia
import skimage
import cv2
import random
import os
from augmentation import MyAugmentationPipeline
from skimage.transform import downscale_local_mean
import glob
from scipy import ndimage

rng = torch.manual_seed(0)
# Defining dataset
class microCT_Dataset(Dataset):

        
    def __init__(self, list_files,train_or_test, HR_patch_size, transform, need_patches):
        
        self.dataset_type = train_or_test
        self.patch_directory=(self.dataset_type+'_input_patches/',self.dataset_type+'_target_patches/')
        
        self.patch_sizeHR = HR_patch_size
        self.patch_sizeLR = int(self.patch_sizeHR*2)# patch_size
        
        self.stepHR = HR_patch_size
        self.stepLR = HR_patch_size
        
        if need_patches == True:
            data_path,target_path = self.generate_patches(list_files)
        else:
            
            data_path= self.patch_directory[0]
            target_path =self.patch_directory[1]
            
        self.data = glob.glob(self.patch_directory[0]+'*')
        self.target = glob.glob(self.patch_directory[1]+'*')
        
        self.transform = transform

        
    def generate_patches(self,list_files):
        if not os.path.exists(self.patch_directory[0]):
            os.makedirs(self.patch_directory[0])
            os.makedirs(self.patch_directory[1])
            
        for f in list_files:
            print ('file name: ', f[0])
            t1 = time.time()
            file_tag = f[0][:2]
            data_path = 'data/' + f[0]
            target_path = 'data/' + f[1]
            # reading data and dropping top and bottom layers
            # these layers are empty becasue of registration 
            data = tifffile.imread(data_path)[10:-10,:,:]
            target = tifffile.imread(target_path)[10:-10,:,:]
            
            print ('reading data shape = ', data.shape)
            print ('reading data max = ', data.max())
            
            # contrast stretching to avoid high intensity artifacts
            data = self.contrast_stretching(data)
            
            #padding inputdata to a proper shape by adding half of the 
            # difference between HR and LR patch size to each side of input
            pd = int((self.patch_sizeLR-self.patch_sizeHR)/2)
            #print ('padding amount = ',pd)
            data = np.pad(data, ((pd,pd), (pd,pd), (pd,pd)), mode='constant') 

            # Data and target into patches
    
            data = self.patchyfy_img(data,self.patch_sizeLR,self.stepLR)
            target = self.patchyfy_img(target,self.patch_sizeHR,self.stepHR)
            
            
            print ('out_of_loader data shape = ', data.shape)
            print ('out_of_loader data max = ', data.max())
    
            print ('out_of_loader target shape = ', target.shape)
            print ('out_of_loader target max = ', target.max())
            

            
            if not data.shape[0] == target.shape[0]:
                print ('the patching is not correct!! STOP')
                return None
            
            for i in range(data.shape[0]):
                
                # the input data was upscaled for image registration.
                # now we downscale it back to original
                down_scaled = downscale_local_mean(data[i], (2,2,2))
                
                tifffile.imwrite(self.patch_directory[0]+file_tag+str(1000+i)+'.tif',down_scaled)
                tifffile.imwrite(self.patch_directory[1]+file_tag+str(1000+i)+'.tif',target[i])
                
        
            t2=time.time()
            print ('Time = ',round((t2-t1)/60),' minutes')
        return self.patch_directory
        
    def patchyfy_img(self,img, ps, step):
        img = patchify(img,(ps, ps, ps) ,  step=step )
        print (img.shape)
        img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2],ps,ps,ps )
        return img

    
    def __len__(self):
        return len(self.data)
    
    def contrast_stretching(self,input_image):
        # Contrast stretching
        # Dropping extreems (artifacts)
        p2, p98 = np.percentile(input_image, (2, 98))
        stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
        return stretched_image
    
    def __getitem__(self, index):
        
        if self.target[index].split('.tif')[0][-5:] == self.data[index].split('.tif')[0][-5:]:
        
            data = tifffile.imread(self.data[index])
            target = torch.from_numpy(tifffile.imread(self.target[index])).float()
            
            if self.transform is not None:
                #Pisel level augmentation always necessary
                random_light = np.ones_like(data)* random.randint(-10, 10)
                data = data + random_light
            
            #Expand LR with linear interpolation to have same dimenssion as HR
            data = ndimage.zoom(data,2, order = 0, prefilter=False, grid_mode=False)
            data = torch.from_numpy(data).float()
            
                    
            if self.transform is not None:
                aug = MyAugmentationPipeline()
                data , target = aug.forward(data, target)
                
            
            
            data= data.unsqueeze(0)
            target = target.unsqueeze(0)
            

        else:
            print ('The patch names or no same')
            return None
        
        # Normalization
        data = data/255
        target = target/255
        
        return data, target


