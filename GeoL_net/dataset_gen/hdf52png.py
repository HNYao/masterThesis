import h5py
import cv2
import numpy as np


def hdf52png(hdf5_path = 'dataset/scene_RGBD_mask/id1/bottle_0003_cola/with_obj/0.hdf5', output_dir='dataset/scene_RGBD_mask/id1/bottle_0003_cola/with_obj/mask.png'):
    category_colors = {
        0: (0, 0, 0),         
        1: (0, 0, 255),   #Obj _ Blue   
        2: (0, 0, 0),   #Table - Black
        3: (0, 0, 0),   #Ground - Black
        4: (0, 255, 0),  # removed obj - Green 
        5: (255, 0, 0) # anchor obj - Red  
    }

    file_path =  hdf5_path
    with h5py.File(file_path, 'r') as hdf5_file:
        segmap = hdf5_file['/category_id_segmaps'][...]
        segmap = segmap.astype(np.int32)
        rgb_image = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
        for category, color in category_colors.items():
            rgb_image[segmap == category] = color
    
        cv2.imwrite(output_dir,rgb_image)
    return rgb_image

def hdf52png_table(hdf5_path, output_dir):
    """
    This function is used to get table mask from category_id seg_maps. only extract table mask
    """
    category_colors = {
        0: (0, 0, 0),         
        1: (0, 0, 0),   #Obj Black   
        2: (255, 255, 255),   #Table - White
        3: (0, 0, 0),   #Ground - Black
        4: (0, 0, 0),  # removed obj - Green 
        5: (0, 0, 0) # anchor obj - Red  
    }

    file_path =  hdf5_path
    with h5py.File(file_path, 'r') as hdf5_file:
        segmap = hdf5_file['/category_id_segmaps'][...]
        segmap = segmap.astype(np.int32)
        rgb_image = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
        for category, color in category_colors.items():
            rgb_image[segmap == category] = color
    
        cv2.imwrite("outputs/img_output/table_mask.png",rgb_image)
    return rgb_image
    

if __name__ == "__main__":
    hdf5_path = "dataset/scene_RGBD_mask_v2/id531_1/phone_0000_normal/no_obj/0.hdf5"
    is_with_obj = hdf5_path.split("/")[-2]
    if is_with_obj == "with_obj":
        png_end = "/mask_with_obj.png"
    else:
        png_end = "/mask_no_obj.png"
    
    output_dir = "dataset/scene_RGBD_mask_v2/id531_1/" + hdf5_path.split("/")[-3] + png_end
    hdf52png_table(hdf5_path, output_dir)
    print(output_dir)
    #hdf52png()