from __future__ import print_function, division
import numpy as np
import cv2
import os

def generate_subimages(image, mask, patch_size, num_xy, num_z):
    """
    Generate sub-images and corresponding masks with the specified patch size using the input 3D image array, 3D mask array, 
    number of patches along x and y dimensions, and number of patches along the z dimension, returning sub-images and masks.
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    depth = np.shape(image)[0]
    patch_width = np.array(patch_size)[1]
    patch_height = np.array(patch_size)[2]
    patch_depth = np.array(patch_size)[0]
    stride_width = (width - patch_width) // num_xy
    stride_height = (height - patch_height) // num_xy
    stride_depth = (depth - patch_depth) // num_z

    if stride_depth >= 1 and stride_width >= 1 and stride_height >= 1:
        step_width = (width - (stride_width * num_xy + patch_width)) // 2
        step_height = (height - (stride_height * num_xy + patch_height)) // 2
        step_depth = (depth - (stride_depth * num_z + patch_depth)) // 2
        image_patches = []
        mask_patches = []
        for z in range(step_depth, num_z * (stride_depth + 1) + step_depth, num_z):
            for x in range(step_width, num_xy * (stride_width + 1) + step_width, num_xy):
                for y in range(step_height, num_xy * (stride_height + 1) + step_height, num_xy):
                    non_zero_count = (mask[z:z + patch_depth, x:x + patch_width, y:y + patch_height] != 0).sum()
                    threshold = patch_depth * patch_width * patch_height / 20.0
                    if non_zero_count > threshold:
                        image_patches.append(image[z:z + patch_depth, x:x + patch_width, y:y + patch_height])
                        mask_patches.append(mask[z:z + patch_depth, x:x + patch_width, y:y + patch_height])
        image_patches_array = np.array(image_patches).reshape((len(image_patches), patch_depth, patch_width, patch_height))
        mask_patches_array = np.array(mask_patches).reshape((len(mask_patches), patch_depth, patch_width, patch_height))
        return image_patches_array, mask_patches_array
    else:
        num_sub_images = 1 * 1 * 1
        image_patches_array = np.zeros(shape=(num_sub_images, patch_depth, patch_width, patch_height), dtype=np.float)
        mask_patches_array = np.zeros(shape=(num_sub_images, patch_depth, patch_width, patch_height), dtype=np.float)
        valid_depth = min(depth, patch_depth)
        valid_width = min(width, patch_width)
        valid_height = min(height, patch_height)
        image_patches_array[0, 0:valid_depth, 0:valid_width, 0:valid_height] = image[0:valid_depth, 0:valid_width, 0:valid_height]
        mask_patches_array[0, 0:valid_depth, 0:valid_width, 0:valid_height] = mask[0:valid_depth, 0:valid_width, 0:valid_height]
        return image_patches_array, mask_patches_array

def create_patches(image, mask, patch_size, num_xy, num_z):
    """
    Create patches from the image and mask using the input 3D image array, 3D mask array, patch size, 
    number of patches along x and y dimensions, and number of patches along the z dimension, returning sub-images and masks.
    """
    image_patches, mask_patches = generate_subimages(image=image, mask=mask, patch_size=patch_size, num_xy=num_xy, num_z=num_z)
    return image_patches, mask_patches

def save_image_and_mask_patches(src_image, mask_image, index, patch_size, num_xy, num_z, train_image_dir, train_mask_dir):
    sub_images, sub_masks = create_patches(src_image, mask_image, patch_size, num_xy, num_z)
    num_samples, patch_depth = np.shape(sub_images)[0], np.shape(sub_images)[1]
    for j in range(num_samples):
        sub_mask = sub_masks.astype(np.float32)
        sub_mask = np.clip(sub_mask, 0, 255).astype('uint8')
        if np.max(sub_mask[j, :, :, :]) == 255:
            image_save_path = os.path.join(train_image_dir, f"{index}_{j}")
            mask_save_path = os.path.join(train_mask_dir, f"{index}_{j}")
            if not os.path.exists(image_save_path) and not os.path.exists(mask_save_path):
                os.makedirs(image_save_path)
                os.makedirs(mask_save_path)
            for z in range(patch_depth):
                image_slice = sub_images[j, z, :, :]
                image_slice = np.clip(image_slice.astype(np.float32), 0, 255).astype('uint8')
                cv2.imwrite(os.path.join(image_save_path, f"{z}.bmp"), image_slice)
                cv2.imwrite(os.path.join(mask_save_path, f"{z}.bmp"), sub_mask[j, z, :, :])

def prepare_3d_training_data(src_dir, mask_dir, train_image_dir, train_mask_dir, num_samples, height, width, patch_size=(16, 256, 256), num_xy=3, num_z=20):
    for sample_index in range(num_samples):
        image_slices = []
        mask_slices = []
        for slice_index in range(len(os.listdir(os.path.join(src_dir, str(sample_index))))):
            image_slice = cv2.imread(os.path.join(src_dir, str(sample_index), f"{slice_index}.bmp"), cv2.IMREAD_GRAYSCALE)
            mask_slice = cv2.imread(os.path.join(mask_dir, str(sample_index), f"{slice_index}.bmp"), cv2.IMREAD_GRAYSCALE)
            image_slices.append(image_slice)
            mask_slices.append(mask_slice)

        image_array = np.array(image_slices).reshape((slice_index + 1, height, width))
        mask_array = np.array(mask_slices).reshape((slice_index + 1, height, width))
        save_image_and_mask_patches(image_array, mask_array, sample_index, patch_size, num_xy, num_z, train_image_dir, train_mask_dir)

def prepare_nodule_detection_training_data():
    height = 512
    width = 512
    num_samples = 56
    src_dir = "process/image/"
    mask_dir = "process/mask/"
    train_image_dir = "segmentation/Image/"
    train_mask_dir = "segmentation/Mask/"
    prepare_3d_training_data(src_dir, mask_dir, train_image_dir, train_mask_dir, num_samples, height, width, (16, 96, 96), 10, 10)

prepare_nodule_detection_training_data()
