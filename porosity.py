import glob
import tifffile
import cv2
import numpy as np



class ice():
    
    def __init__(self,img, name, starting_layer,ending_layer):
        self.img = img
        core_name = file_name[:3]
        print (core_name)
        self.sorting_depth = int(name.split('Bag')[-1].split('_')[0])
        self.starting_layer = starting_layer
        self.ending_layer = ending_layer
        if core_name == 'B45':
            self.single_core_depth = name.split('Bag')[-1].split('_')[0]
            self.left_depth = None
            self.right_depth = None
            
        elif core_name == 'B40':
            self.left_depth = int(name.split('Bag')[-1].split('_')[0])
            self.right_depth = int(name.split('Bag')[-1].split('_')[1])
            self.single_core_depth = None
        else:
            print ('**** Error ***** The core name is not found')
        
        print ('sorting_depth',self.sorting_depth)
        print('left_depth', self.left_depth)
        print('right_depth', self.right_depth) 
        print('single_core_depth', self.single_core_depth)
        
            
    def get_ice_part(self):
        """
        kernel_size_dilute = 2 for bubble ice
        kernel_size_dilute = 20 for compressed snow
        """
        thresh1=0.5
        kernel_size_erode=50
        
        final_mask = np.zeros_like(self.img)
        kernel_dilute = np.ones((self.kernel_size_dilute,self.kernel_size_dilute),np.uint8)
    
        for i in range(self.img.shape[0]):
            an_img = self.img[i]
            # Apply a binary threshold to create a binary image
            _, binary = cv2.threshold(an_img, thresh1, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            diluted = cv2.dilate(binary,kernel_dilute,iterations = 1)
            contours, _ = cv2.findContours(diluted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
            mask = np.zeros_like(binary) # contains ice
    
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Adjust this threshold based on the size of the ice pieces
                    cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
            kernel_erode = np.ones((kernel_size_erode,kernel_size_erode),np.uint8)
            erosion_mask = cv2.erode(mask,kernel_erode,iterations = 1)
            if erosion_mask.sum()<1e3:
                continue
            final_mask[i] = erosion_mask
        return final_mask.astype('uint8')
            
    
    
    
    
    def separate_cores(self):
    
        """
        This function takes the binariezed mask and splite two ice cores into two regions
        labeling 1 for left ice core
        labeling 2 for right ice core
        """
        from skimage.measure import label, regionprops
        
        labelled, tnb = label(self.mask, connectivity=2, background=0, return_num=True)
        rg = regionprops (labelled)
        print ('number of detected regions =', len(rg))
        self.number_of_regions = len(rg)
        ice_mask = np.zeros_like(self.mask)
        
        if self.mask.sum()<100:
            return ice_mask
        
        center_x = ice_mask.shape[2]//2
        for r in rg:
            if r.centroid[2] < center_x:
                ice_mask[labelled == r.label] = 1
            else:
                ice_mask[labelled == r.label] = 2
                
        return ice_mask
        
    
    def porosity_single_ice(self):
        
        if self.sorting_depth >= 20:
            self.kernel_size_dilute = 2
        else:
            self.kernel_size_dilute = 20
        
        mask = self.get_ice_part()
            
        region = self.img[mask == 1]
        self.single_vol = len(region)
        air = len(region[region==0])
        self.single_porosity = round((100*air)/self.single_vol,2)
        
    def porosity_double_ice (self):   
        if self.sorting_depth >= 20:
            self.kernel_size_dilute = 2
        else:
            self.kernel_size_dilute = 20
        
        self.mask = self.get_ice_part()
        self.mask = self.separate_cores()
        
        region = self.img[self.mask == 1]
        self.left_vol = len(region)
        air = len(region[region==0])
        self.left_porosity = round((100*air)/self.left_vol,2)
        
        
        
        region = self.img[self.mask == 2]
        self.right_vol = len(region)
        air = len(region[region==0])
        self.right_porosity = round((100*air)/self.right_vol,2)
        



def worker(img_chunk, file_name,starting_layer, ending_layer):
    ice_1 = ice(img_chunk, file_name,starting_layer, ending_layer)
    ice_1.porosity_double_ice()
    print ('left_porosity',ice_1.left_porosity)
    print ('left_volume', ice_1.left_vol)
        
    print ('right_porosity',ice_1.right_porosity)
    print ('right_volume', ice_1.right_vol)
    
    return




        
files = glob.glob('../B40_SR_seg/*.tif')
print (files)
print ('***************************')
for f in files:
    step = 10
    file_name = f.split('/')[-1].split('\\')[-1].split('.tif')[0]
    img = tifffile.imread(f)
    for i in range(0,img.shape[0],step):
        starting_layer = i
        ending_layer = i+step
        
        if i+step >= img.shape[0]:
            ending_layer = -1
            
        img_chunk = img[starting_layer:ending_layer]
        w = worker(img_chunk, file_name,starting_layer, ending_layer)

        break

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    