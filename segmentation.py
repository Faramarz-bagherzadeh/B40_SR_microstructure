

def get_ice_part(image,skip,thresh1, kernel_size):
    final_mask = np.zeros_like(image)
    for i in range(0,image.shape[0]-skip,skip):
        img = image[i]
        if img.sum()<4e6: # passing to next image if it is mostly black
            continue
            
        # Apply a binary threshold to create a binary image
        _, binary = cv2.threshold(img, thresh1, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        mask = np.zeros_like(binary) # contains ice

        for cnt in contours[:]:
            area = cv2.contourArea(cnt)
            if area > 100:  # Adjust this threshold based on the size of the ice pieces
                cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
                
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        erosion_mask = cv2.erode(mask,kernel,iterations = 1)       
        if erosion_mask.sum()<1e3:
            continue 
        final_mask[i] = erosion_mask
    
    return final_mask




def contrast_stretching(input_image):
    import skimage
    #Contrast stretching
    #Dropping extreems (artifacts)
    p2, p98 = np.percentile(input_image, (2, 98))
    stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
    return stretched_image.astype('uint8')

        

def binary_seg_kMeans(img):
    from sklearn.cluster import KMeans
    from joblib import parallel_backend

    
    binary = np.zeros_like(img)
    mask = get_ice_part(img, skip=20,thresh1=15,kernel_size=100)
    pixels = img[mask==1].reshape(-1, 1)
    with parallel_backend('threading', n_jobs=-1):
        kmeans = KMeans(n_init=4, n_clusters=2,)
        kmeans.fit(pixels)
        
    centers = kmeans.cluster_centers_
    thresh = (centers[0] + centers[1])/2
    binary[img > thresh] = 1
    
    return binary, round(thresh[0])



def segmentation_function(image, batch):
    thresh_list=[]
    segmented_img = np.zeros_like(image)
    for s in range (0,image.shape[0],batch):
        print ('steps = ',s,s+batch)
        if s+batch > image.shape[0]:
            img = image[s:,:,:]
        else:
            img = image[s:s+batch,:,:]
                
        img = contrast_stretching(img)
        binary, k_thresh = binary_seg_kMeans(img)
        thresh_list.append(k_thresh)

        if s+batch > image.shape[0]:
            segmented_img[s:,:,:] = binary
        else:
            segmented_img[s:s+batch,:,:]= binary        
    
    print (thresh_list)
    return segmented_img.astype('uint8')


if __name__ == "__main__":
    import cv2
    import numpy as np
    import tifffile
    import glob
    import time
    
    batch = 2000
    
    paths = glob.glob('../B40_SR_gray/*.tif')
    print(paths)
    print ('number of files =' ,len(paths))
    
    
    
    for f in paths:
        t1 = time.time()
        data = tifffile.imread(f)
        name = f.split('/')[-1].split('.')[0]
        print ('*********************************************')
        print ('file name = ', name)
        print ('shape = ',data.shape)
        segmented_data = segmentation_function(data, batch)

        t2= time.time()
        print ('Time (min) =', round((t2-t1)/60))
        
        
        tifffile.imwrite('../B40_SR_seg/'+name+'_segmented_.tif', segmented_data)
    



















