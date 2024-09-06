
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
import numpy as np
from torchvision import transforms
import dataloader
import tifffile
import skimage
torch.manual_seed(17)
import random
import time
from patchify import patchify, unpatchify
import math
import SRResnet

# Set the hyperparameters
num_epochs = 4
learning_rate = 5e-4
patch_size = 64
batch_size = 8
n_chanels = 8
num_residual_blocks = 2


if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {device_count}")
    for i in range(device_count):
        device = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device}")
else:
    print("CUDA is not available on this machine.")
    
    
# while choosing patch size need to consider the test volume size (depth) 
# in such way to include at least one patch 


# Set the paths to data and target files
train_files = [
    ('01_Substack (7882-8277)_B45_bag5_.tif','registered_image01_3dCT_B45_bag5_100mm.tif'),
    #('02_Substack (6267-6662)_B40_bag12_13.tif','registered_image02_3dCT_B40_bag12_13_300mm.tif' ),
    ('03_Substack (6241-6635)_B40_bag22_23.tif', 'registered_image03_3dCT_B40_bag22_23_300mmLUT.tif' ),
    ('04_Substack (7285-7546)_B40_bag32_33.tif','registered_image04_3dCT_B40_bag32_33_200mm_Substack(396-1440).tif' ),
    ('05_Substack (6292-6686)_B40_bag46_47.tif','registered_image05_3dCT_B40_bag46_47_300mm.tif'),
    #('06_Substack (8055-8449)_B40_bag56_57.tif','registered_image06_3dCT_B40_bag56_57_100mm.tif' ),
    ('07_Substack (5525-5920)_B40_bag66_67.tif', 'registered_image07_3dCT_B40_bag66_67_400mm.tif' ),
    ('08_Substack (3424-3818)_B40_bag86_87.tif', 'registered_image08_3dCT_B40_bag86_87_640mm.tif' ),
    ('09_Substack (3098-3490)_B40_bag96_97.tif', 'registered_image09_3dCT_B40_bag96_97_675mm.tif' ),
    #('10_Substack (4268-4663)_B40_bag108_109.tif','registered_image10_3dCT_B40_bag108_109_538mm.tif'),
    ('11_Substack (305-698)_B40_bag126_127.tif', 'registered_image11_3dCT_B40_bag126_127_1002mm.tif')
    ]

test_files = [
    #('01_Substack (7882-8277)_B45_bag5_.tif','registered_image01_3dCT_B45_bag5_100mm.tif'),
    ('02_Substack (6267-6662)_B40_bag12_13.tif','registered_image02_3dCT_B40_bag12_13_300mm.tif' ),
    #('03_Substack (6241-6635)_B40_bag22_23.tif', 'registered_image03_3dCT_B40_bag22_23_300mmLUT.tif' ),
    #('04_Substack (7285-7546)_B40_bag32_33.tif','registered_image04_3dCT_B40_bag32_33_200mm_Substack(396-1440).tif' ),
    #('05_Substack (6292-6686)_B40_bag46_47.tif','registered_image05_3dCT_B40_bag46_47_300mm.tif'),
    ('06_Substack (8055-8449)_B40_bag56_57.tif','registered_image06_3dCT_B40_bag56_57_100mm.tif' ),
    #('07_Substack (5525-5920)_B40_bag66_67.tif', 'registered_image07_3dCT_B40_bag66_67_400mm.tif' ),
    #('08_Substack (3424-3818)_B40_bag86_87.tif', 'registered_image08_3dCT_B40_bag86_87_640mm.tif' ),
    #('09_Substack (3098-3490)_B40_bag96_97.tif', 'registered_image09_3dCT_B40_bag96_97_675mm.tif' ),
    ('10_Substack (4268-4663)_B40_bag108_109.tif','registered_image10_3dCT_B40_bag108_109_538mm.tif'),
    #('11_Substack (305-698)_B40_bag126_127.tif', 'registered_image11_3dCT_B40_bag126_127_1002mm.tif')
    ]

# If need to generate pathes for the firs time then put true for the need_patches
train_dataset = dataloader.microCT_Dataset(train_files,train_or_test = 'train',
                                         HR_patch_size=patch_size,
                                         transform = True,
                                          need_patches=True)


# If need to generate pathes for the firs time then put true for the need_patches
test_dataset = dataloader.microCT_Dataset(test_files,train_or_test = 'test',
                                         HR_patch_size=patch_size,
                                         transform = None,
                                          need_patches=True)

train_data_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

test_data_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False)
                              
                              
# Training function
def train(model, train_data_loader, test_data_loader, num_epochs, learning_rate):
    #checking differetn criterions 
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    
    for epoch in range(num_epochs):
        t1 = time.time()
        model.train()
        train_loss = 0.0
        counter = 1
        stop = 10e8 # change this if training on smaller data is needed to check if everything is working
        for data, target in train_data_loader:
            if data.max()>1.5:
                print ('Input data has issue !!!!')
                break
            print ('input shape {}  '.format(data.shape),'Training progress for epoch {}'.format(epoch+1),
                   counter*100 //len(train_data_loader),'%', end = '\r')
            counter +=1
            if counter>stop:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            #If using a cirterion uncomment this line and comment out the Kornia loss
            loss = criterion(output, target)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        #scheduler.step()
        t2 = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss / len(train_data_loader):.4f}, Time (min) = {round((t2-t1)/60)}  ")

        #Calculate Metrics on test set
        model.eval()
        with torch.no_grad():
            PNSR= []
            RMSE = []
            SSI = []
            t_counter = 1
            for data, target in test_data_loader:
                if data.max()>1.5:
                    print ('Input data has issue of Normalization !!!!')
                    break 
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = output.cpu().numpy()#.flatten()
                target= target.cpu().numpy()#.flatten()
                PNSR.append(skimage.metrics.peak_signal_noise_ratio(target, output, data_range=None))
                RMSE.append(math.sqrt(skimage.metrics.mean_squared_error(target, output)))
                SSI.append(skimage.metrics.structural_similarity(target, output, win_size=7, gradient=False,data_range=None, channel_axis=None,gaussian_weights=False, full=False,))
         
                t_counter +=1
                print ('testing progress for epoch {}% _ '.format(epoch+1),t_counter*100 //len(test_data_loader), end = '\r')

            print ('epoch = {}'. format(epoch))
            print(f"RMSE: {np.array(RMSE).mean()}")
            print(f"SSI: {np.array(SSI).mean()}")
            print(f"PNSR: {np.array(PNSR).mean()}")



# Create an instance of the model
model = SRResnet.SRResNet(num_blocks = num_residual_blocks, n_chanels = n_chanels)

import warnings
warnings.filterwarnings("ignore")
# Train the model
train(model, train_data_loader, test_data_loader, num_epochs, learning_rate)


#save model
def save_checkpoint(state, filename="my_SRResnet_{}_checkpoint.pth.tar".format(num_residual_blocks)):
    print("=> Saving checkpoint after {} epochs".format(num_epochs))
    torch.save(state, filename)

checkpoint = {"state_dict": model.state_dict()}
save_checkpoint(checkpoint)
print ('***** SRResNet with {} Res blocks ******'.format(num_residual_blocks))
