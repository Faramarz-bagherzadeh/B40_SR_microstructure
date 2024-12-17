import numpy as np
from skimage.measure import label, regionprops

def classify_pores(binary_image):
    """
    Classify pores in a 3D binary image into three groups:
    
    Group 1: Pores with a bounding box dimension >= the first dimension of the image.
    Group 2: Pores connected to a face of the image but not in Group 1.
    Group 3: Pores totally isolated.
    
    Parameters:
        binary_image (numpy.ndarray): A 3D binary image where solid is 1 and pore is 0.

    Returns:
        dict: A dictionary with group keys and corresponding pixel sums.
    """
    # Label connected components in the pore space (binary 0)
    inverted_image = np.logical_not(binary_image)  # Invert the binary image (1 -> 0, 0 -> 1)
    labeled_image, num_features = label(inverted_image, connectivity=3, return_num=True)

    group_1_sum = 0
    group_2_sum = 0
    group_3_sum = 0
    volume = binary_image.shape[0] * binary_image.shape[1] * binary_image.shape[2] 
    image_shape = binary_image.shape

    # Identify pores in each group
    for region in regionprops(labeled_image):
        bbox = region.bbox
        bbox_size = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]  # 3D bounding box sizes

        # Group 1: Bounding box size >= first dimension of the image
        if any(dim >= image_shape[0] for dim in bbox_size):
            group_1_sum += region.area
        
        else:
            # Determine if the region touches any face of the image
            min_z, min_y, min_x, max_z, max_y, max_x = bbox
            touches_face = (
                min_z == 0 or min_y == 0 or min_x == 0 or
                max_z == image_shape[0] or
                max_y == image_shape[1] or
                max_x == image_shape[2]
            )

            if touches_face:
                # Group 2: Touches a face but not in Group 1
                group_2_sum += region.area
            else:
                # Group 3: Completely isolated
                group_3_sum += region.area

    return round(100*group_1_sum/volume ,2), round(100*group_2_sum/volume,2) , round(100*group_3_sum/volume,2)


    






