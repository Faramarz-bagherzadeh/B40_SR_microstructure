import glob
import tifffile
import cv2
import numpy as np
from multiprocessing import Pool
import math
import pandas as pd
from skimage.measure import label, regionprops
from skimage import measure
import pumapy as puma
import microstructure_parameters as mp
import openpnm as op
import porespy as ps
from openporosity import classify_pores

class ice():
    def __init__(self,img, name, starting_layer,ending_layer,sample_size):
        #tifffile.imwrite('samples/img.tif',img)
        self.sample_size = sample_size
        self.resolution = 60e-6
        self.pixel_volume = (0.006) ** 3
        self.volume_cm = self.sample_size**3 * self.pixel_volume
        self.img = img
        self.name = name.split('_')[1]
        self.core_name =name[:3]
        #print (self.core_name)
        self.sorting_depth = int(name.split('Bag')[-1].split('_')[0])
        self.starting_layer = starting_layer
        self.ending_layer = ending_layer
          # cm³
       # print ('core name = ',self.core_name)

        if self.core_name == 'B45':
            self.single_core_depth = name.split('Bag')[-1].split('_')[0]
            self.left_depth = None
            self.right_depth = None

        elif self.core_name == 'B40' and self.name.split('Bag')[-1].split('_')[0]=='7':
            self.single_core_depth = name.split('Bag')[-1].split('_')[0]
            self.left_depth = None
            self.right_depth = None


        elif self.core_name == 'B40':
            self.left_depth = int(name.split('Bag')[-1].split('_')[0])
            self.right_depth = int(name.split('Bag')[-1].split('_')[1])
            self.single_core_depth = None
        else:
            print ('**** Error ***** The core name is not found')

        #print ('sorting_depth',self.sorting_depth)
        #print('left_depth', self.left_depth)
        #print('right_depth', self.right_depth)
        #print('single_core_depth', self.single_core_depth)


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
                if area > 10:  # Adjust this threshold based on the size of the ice pieces
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
        ice_mask = np.zeros_like(self.mask)
        labelled, tnb = label(self.mask, connectivity=2, background=0, return_num=True)

        try: # if there are no regions then skip the error and return mask of zeros
            rg = regionprops (labelled)
        except:
            self.number_of_regions = 0
            return ice_mask

        #print ('number of detected regions =', len(rg))
        self.number_of_regions = len(rg)


        if self.mask.sum()<100:
            return ice_mask

        center_x = ice_mask.shape[2]//2
        for r in rg:
            if r.centroid[2] < center_x:
                ice_mask[labelled == r.label] = 1
            else:
                ice_mask[labelled == r.label] = 2

        return ice_mask

    def set_sample_volume(self, mask_1, side):
        """
        For left side then side = 1 and Right side = 2
        This function takes a data with z layers and then
        gives sample output of z*z*z from center of core
        """

        mask = np.zeros_like(mask_1)
        mask [mask_1 ==side]=1
        half_edge = self.img.shape[0]// 2
        sample = np.zeros_like(mask)
        try:
            rg = regionprops (mask)
            center = rg[0].centroid
        except:
            center = tuple((np.array(self.img.shape) / 2).astype(int))

        x_start = int( center[2] - half_edge)
        x_end = int(center[2] + half_edge )

        y_start = int(center[1] - half_edge)
        y_end = int(center[1] + half_edge )

        sample[:, y_start:y_end, x_start:x_end] = 1

        print (x_start,x_end,y_start,y_end)
        #if len(sample[(mask==1) & (sample==1)]) < sample.sum():
            # checking if any crack is inside the sample
        #    print ('Sample not usable')
        #    return None

        output_data = self.img[:, y_start:y_end, x_start:x_end]

        return output_data

    def puma_specific_area(self,binary):
        ws = puma.Workspace.from_array(binary)
        ws.voxel_length = self.resolution
        area_us, specific_area_us = puma.compute_surface_area(ws, cutoff=(1, 1))

        return specific_area_us

    def puma_mean_intercept_length(self,binary):

        ws = puma.Workspace.from_array(binary)
        ws.voxel_length = self.resolution
        mil = puma.compute_mean_intercept_length(ws, void_cutoff=(1, 1))
        mil = np.array(mil)
        mil = mil * 1000 # change unit to mm
        mil = np.round(mil, 3)
        # changing milx and milz because of the difference between numpy and puma
        mil = (mil[2],mil[1],mil[0])
        return mil


    def puma_permeability (self,img):
        k=[]
        return [0,0,0]
        inputs = [(img, self.resolution,'zmin','zmax'),(img, self.resolution,'ymin','ymax'),(img, self.resolution,'xmin','xmax')]
        for inp in inputs:
            try:
            # Permeability of X and Z are replaced because diffrence between numpy and porspy indexing
                result = mp.calculate_permeability(*inp)
                k.append(result)
            except:
                k.append(-100)
        return k


    def puma_tortuosity (self,img):
        tau=[]
        inputs = [(img, self.resolution,'zmin','zmax'),(img, self.resolution,'ymin','ymax'),(img, self.resolution,'xmin','xmax')]
        for inp in inputs:
            try:
                result = mp.calculate_tortuosity(*inp)
                #result = 0
                tau.append(result)
            except:
                tau.append(-100)
        print ('***  Tau = ',tau)
        return tau


    def microstructure_single(self):
        print ('Single core activated')
        self.number_of_regions = 1
        if self.sorting_depth >= 20:
            self.kernel_size_dilute = 2
        else:
            self.kernel_size_dilute = 2 #20

        self.mask = self.get_ice_part()

        self.single_sample = self.set_sample_volume(self.mask, side=1)
        #To make the network should make the image to bool
        #The pore space (void) should be True to allow flow going through it
        bool_img = self.single_sample==0

        if self.single_sample is not None:
            self.single_vol =  self.single_sample.size
            self.single_porosity = round(len(self.single_sample[self.single_sample==0])*100/(self.single_vol+1),2)
            self.single_pores = classify_pores(self.single_sample)
            self.single_euler = round(measure.euler_number(self.single_sample, connectivity=1)/self.volume_cm,2)
            self.single_SSA = self.puma_specific_area(self.single_sample)
            self.single_MIL= self.puma_mean_intercept_length(self.single_sample)
            self.single_Perm = self.puma_permeability(bool_img)
            self.single_Tort = self.puma_tortuosity(bool_img)
            self.single_sph_ice_cluster = mp.spherical_ice_cluster(self.single_sample)
            self.single_sk = mp.skeleton_metrics(bool_img, self.resolution)

    def microstructure_double_ice (self):
        from skimage import measure

        if self.sorting_depth >= 20:
            self.kernel_size_dilute = 2
        else:
            self.kernel_size_dilute = 20

        self.mask = self.get_ice_part()
        self.mask = self.separate_cores()

        self.left_sample = self.set_sample_volume(self.mask, side=1)
        bool_img =self.left_sample==0

        #self.left_sample_network=ps.networks.snow2(bool_img, sigma=0.4, r_max=4, voxel_size=self.resolution,accuracy='high')
       # tifffile.imwrite('samples/left.tif',self.left_sample)
        if self.left_sample is not None:
            self.left_vol =  self.left_sample.size
            self.left_porosity = round(len(self.left_sample[self.left_sample==0])*100/(self.left_vol+1),2)
            self.left_pores = classify_pores(self.left_sample)
            self.left_euler = round(measure.euler_number(self.left_sample, connectivity=1)/self.volume_cm ,2)
            self.left_SSA = self.puma_specific_area(self.left_sample)
            self.left_MIL= self.puma_mean_intercept_length(self.left_sample)
            self.left_Perm = self.puma_permeability(bool_img)
            self.left_Tort = self.puma_tortuosity(bool_img)
            self.left_sph_ice_cluster = mp.spherical_ice_cluster(self.left_sample )
            self.left_sk = mp.skeleton_metrics(bool_img, self.resolution)


        self.right_sample = self.set_sample_volume(self.mask, side=2)
        bool_img =self.right_sample==0

        #self.right_sample_network=ps.networks.snow2(bool_img, sigma=0.4, r_max=4, voxel_size=self.resolution,accuracy='high')
       # tifffile.imwrite('samples/right.tif',self.right_sample)
        if self.right_sample is not None:
            self.right_vol =  self.right_sample.size
            self.right_porosity = round(len(self.right_sample[self.right_sample==0])*100/(self.right_vol+1),2)
            self.right_pores = classify_pores(self.right_sample)
            self.right_euler = round(measure.euler_number(self.right_sample, connectivity=1)/self.volume_cm ,2)
            self.right_SSA = self.puma_specific_area(self.right_sample)
            self.right_MIL= self.puma_mean_intercept_length(self.right_sample)
            self.right_Perm = self.puma_permeability(bool_img)
            self.right_Tort = self.puma_tortuosity(bool_img)
            self.right_sph_ice_cluster = mp.spherical_ice_cluster(self.right_sample)
            self.right_sk = mp.skeleton_metrics(bool_img, self.resolution)


def worker(aninput):
    img_chunk, file_name,starting_layer, ending_layer,sample_size = aninput
    results = []
    ice_1 = ice(img_chunk, file_name,starting_layer, ending_layer,sample_size)

    print ('worker started')
    if ice_1.single_core_depth == None:
        try:
            print ('Double core image activated')
            ice_1.microstructure_double_ice()
        except:
            print ('****************  AN ERROR in Double Microstrutcture Happened ************')
            return None

        #print ('left_porosity',ice_1.left_porosity)
        #print ('left_volume', ice_1.left_vol)

        if ice_1.left_sample is not None:
            results.append((

                ice_1.name,
                ice_1.number_of_regions,
                ice_1.starting_layer,
                ice_1.ending_layer,
                ice_1.left_depth,
                ice_1.left_vol,
                ice_1.left_porosity,
                ice_1.left_pores[0],
                ice_1.left_pores[1],
                ice_1.left_pores[2],
                ice_1.left_euler,
                ice_1.left_SSA,
                ice_1.left_MIL[0],
                ice_1.left_MIL[1],
                ice_1.left_MIL[2],
                ice_1.left_Perm[0],
                ice_1.left_Perm[1],
                ice_1.left_Perm[2],
                ice_1.left_Tort[0],
                ice_1.left_Tort[1],
                ice_1.left_Tort[2],
                ice_1.left_sph_ice_cluster,
                ice_1.left_sk['num_pores'],ice_1.left_sk['num_throats'],
                ice_1.left_sk['coordination_number'],ice_1.left_sk['std_coordination_number'],
                ice_1.left_sk['avg_pore_volume'],ice_1.left_sk['avg_pore_diameter'],ice_1.left_sk['avg_throat_diameter'],
                ice_1.left_sk['avg_throat_length'],ice_1.left_sk['max_connections'],ice_1.left_sk['median_connections'],
                ice_1.left_sk['num_cluster'],ice_1.left_sk['max_cluster_size'],ice_1.left_sk['avg_cluster_size'],
                ice_1.left_sk['avg_pore_surface_area'],ice_1.left_sk['avg_throat_area']

                ))

        #print ('right_porosity',ice_1.right_porosity)
        #print ('right_volume', ice_1.right_vol)
        if ice_1.right_sample is not None:
            results.append((
                ice_1.name,
                ice_1.number_of_regions,
                ice_1.starting_layer,
                ice_1.ending_layer,
                ice_1.right_depth,
                ice_1.right_vol,
                ice_1.right_porosity,
                ice_1.right_pores[0],
                ice_1.right_pores[1],
                ice_1.right_pores[2],
                ice_1.right_euler,
                ice_1.right_SSA,
                ice_1.right_MIL[0],
                ice_1.right_MIL[1],
                ice_1.right_MIL[2],
                ice_1.right_Perm[0],
                ice_1.right_Perm[1],
                ice_1.right_Perm[2],
                ice_1.right_Tort[0],
                ice_1.right_Tort[1],
                ice_1.right_Tort[2],
                ice_1.right_sph_ice_cluster,
                ice_1.right_sk['num_pores'],ice_1.right_sk['num_throats'],
                ice_1.right_sk['coordination_number'],ice_1.right_sk['std_coordination_number'],
                ice_1.right_sk['avg_pore_volume'],ice_1.right_sk['avg_pore_diameter'],ice_1.right_sk['avg_throat_diameter'],
                ice_1.right_sk['avg_throat_length'],ice_1.right_sk['max_connections'],ice_1.right_sk['median_connections'],
                ice_1.right_sk['num_cluster'],ice_1.right_sk['max_cluster_size'],ice_1.right_sk['avg_cluster_size'],
                ice_1.right_sk['avg_pore_surface_area'],ice_1.right_sk['avg_throat_area']
                ))
            print('Right side attached!!1')
        return results
    else:
        try:
            ice_1.microstructure_single()
        except:
            return None

        if ice_1.single_sample is not None:
            results.append((
                ice_1.name,
                ice_1.number_of_regions,
                ice_1.starting_layer,
                ice_1.ending_layer,
                ice_1.single_core_depth,
                ice_1.single_vol,
                ice_1.single_porosity,
                ice_1.single_pores[0],
                ice_1.single_pores[1],
                ice_1.single_pores[2],
                ice_1.single_euler,
                ice_1.single_SSA,
                ice_1.single_MIL[0],
                ice_1.single_MIL[1],
                ice_1.single_MIL[2],
                ice_1.single_Perm[0],
                ice_1.single_Perm[1],
                ice_1.single_Perm[2],
                ice_1.single_Tort[0],
                ice_1.single_Tort[1],
                ice_1.single_Tort[2],
                ice_1.single_sph_ice_cluster,
                ice_1.single_sk['num_pores'],ice_1.single_sk['num_throats'],
                ice_1.single_sk['coordination_number'],ice_1.single_sk['std_coordination_number'],
                ice_1.single_sk['avg_pore_volume'],ice_1.single_sk['avg_pore_diameter'],ice_1.single_sk['avg_throat_diameter'],
                ice_1.single_sk['avg_throat_length'],ice_1.single_sk['max_connections'],ice_1.single_sk['median_connections'],
                ice_1.single_sk['num_cluster'],ice_1.single_sk['max_cluster_size'],ice_1.single_sk['avg_cluster_size'],
                ice_1.single_sk['avg_pore_surface_area'],ice_1.single_sk['avg_throat_area']
                ))

    return results

def chunk_data(data,file_name,step ,sample_size):

    # Function to split the input data into chunks
    # Split data into chunks of chunk_size
    for i in range(0,data.shape[0],step):
        starting_layer = i
        ending_layer = i+sample_size

        if i+sample_size >= data.shape[0]:
           # ending_layer = -1
           break

        yield [data[starting_layer:ending_layer],file_name,starting_layer,ending_layer,sample_size]


# Main function to process input using multiprocessing
def process_data(file_name,step, num_workers,sample_size):
    global data
    # Split the input data into chunks
    chunks = list(chunk_data(data,file_name,step,sample_size))

    del data
    print ('number of data chunks = ', len(chunks))
    print ('shape of image per chunk: ', chunks[0][0].shape)
    with Pool(processes=num_workers) as pool:
        # Use pool.map to send each chunk to a worker
        pool_results = pool.map(worker, chunks)
        chunk_results = [result for result in pool_results if result is not None]

    return chunk_results






if __name__ == "__main__":
    import multiprocessing
    import sys
    index = int(sys.argv[1])

    num_cores = multiprocessing.cpu_count()
    print ('number of cpus = ', num_cores)
    import time
    num_workers = 10  # Number of workers you want to use
    sample_size = 400
    overlap = 200
    step_size = sample_size - overlap
    files = glob.glob('../B40_SR_seg/*.tif')
    def dir_sorting(e):
        return int(e.split('Bag')[1].split('_')[0])
    files.sort(key=dir_sorting)
   # print (files)
    print ('**************************')
    print ('number of files = ', len(files))
    print ('***************************')
    #for f in [files[i] for i in range(2,64,2) ]:
    for f in files[index:index +1]:

        t1=time.time()
        file_name = f.split('/')[-1].split('\\')[-1].split('.tif')[0]
        print ('file name : ', file_name)
        data = tifffile.imread(f)[:]
        print ('shape of data = ' , data.shape)

        result_list = process_data(file_name,step_size, num_workers,sample_size)
       # del data
        print (result_list)
        # Flatten the nested list of tuples into a single list
        flattened_data = [item for sublist in result_list for item in sublist]
        print('HERE is FLAT')
        print(flattened_data)
        # Create a DataFrame with specified column names
        df = pd.DataFrame(flattened_data, columns=['name', 'number_of_regions',
                                                   'starting_layer', 'ending_layer',
                                                   'depth', 'sample_volume',
                                                   'porosity','open_pores','cuted_pores','isolated_pores',
                                                   'euler_density','SSA',
                                                   'MIL_x','MIL_y','MIL_z',
                                                   'Perm_x','Perm_y','Perm_z',
                                                   'Tort_x','Tort_y','Tort_z','sph_ice_cluster',
                                                   'num_pores','num_throats','coordination_number','std_coordination_number',
                                                   'avg_pore_volume','avg_pore_diameter','avg_throat_diameter',
                                                   'avg_throat_length','max_connections','median_connections',
                                                   'num_cluster','max_cluster_size','avg_cluster_size',
                                                   'avg_pore_surface_area','avg_throat_area'])

        df.to_excel('output/{}.xlsx'.format(file_name))
        print ('Saved ... porosity_{}'.format(file_name))
        t2 = time.time()
        print ('Time (minutes) = ',round((t2-t1)/60))