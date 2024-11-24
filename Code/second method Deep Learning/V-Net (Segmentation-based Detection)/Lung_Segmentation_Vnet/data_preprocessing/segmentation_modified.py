from __future__ import print_function, division
import numpy as np
import cv2
import os

def generate_subimages(image, mask=None, patch_size=(16, 256, 256), num_patches_xy=3, num_patches_z=20):
    """
    Generate sub-images and corresponding masks with the specified patch size using the input 3D image array, 3D mask array, 
    number of patches along x and y dimensions, and number of patches along the z dimension, returning sub-images and masks.
    """
    width = np.shape(image)[2]
    height = np.shape(image)[1]
    depth = np.shape(image)[0]
    patch_width = patch_size[2]
    patch_height = patch_size[1]
    patch_depth = patch_size[0]
    
    stride_width = max((width - patch_width) // num_patches_xy, 1)
    stride_height = max((height - patch_height) // num_patches_xy, 1)
    stride_depth = max((depth - patch_depth) // num_patches_z, 1)

    image_patches = []
    mask_patches = []

    for z in range(0, depth - patch_depth + 1, stride_depth):
        for x in range(0, width - patch_width + 1, stride_width):
            for y in range(0, height - patch_height + 1, stride_height):
                image_patch = image[z:z + patch_depth, y:y + patch_height, x:x + patch_width]
                image_patches.append(image_patch)
                if mask is not None:
                    mask_patch = mask[z:z + patch_depth, y:y + patch_height, x:x + patch_width]
                    mask_patches.append(mask_patch)

    image_patches_array = np.array(image_patches)
    if mask is not None:
        mask_patches_array = np.array(mask_patches)
        return image_patches_array, mask_patches_array
    else:
        return image_patches_array, None

def create_patches(image, mask, patch_size, num_patches_xy, num_patches_z):
    """
    Create patches from the image and mask using the input 3D image array, 3D mask array, patch size, 
    number of patches along x and y dimensions, and number of patches along the z dimension, returning sub-images and masks.
    """
    image_patches, mask_patches = generate_subimages(image=image, mask=mask, patch_size=patch_size,
                                                     num_patches_xy=num_patches_xy, num_patches_z=num_patches_z)
    return image_patches, mask_patches

def save_image_and_mask_patches(image_array, mask_array, sample_index, patch_size, num_patches_xy, num_patches_z, image_output_dir, mask_output_dir):
    sub_images, sub_masks = create_patches(image_array, mask_array, patch_size=patch_size, num_patches_xy=num_patches_xy, num_patches_z=num_patches_z)
    num_samples = np.shape(sub_images)[0]
    patch_depth = np.shape(sub_images)[1]
    for sample_num in range(num_samples):
        if sub_masks is not None:
            sub_mask = sub_masks.astype(np.float32)
            sub_mask = np.clip(sub_mask, 0, 255).astype('uint8')
        image_save_path = os.path.join(image_output_dir, f"{sample_index}_{sample_num}")
        if sub_masks is not None:
            mask_save_path = os.path.join(mask_output_dir, f"{sample_index}_{sample_num}")
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if sub_masks is not None and not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        for z in range(patch_depth):
            image_slice = sub_images[sample_num, z, :, :]
            image_slice = image_slice.astype(np.float32)
            image_slice = np.clip(image_slice, 0, 255).astype('uint8')
            cv2.imwrite(os.path.join(image_save_path, f"{z}.bmp"), image_slice)
            if sub_masks is not None:
                cv2.imwrite(os.path.join(mask_save_path, f"{z}.bmp"), sub_mask[sample_num, z, :, :])

def prepare_3d_training_data(image_dir, mask_dir, image_output_dir, mask_output_dir, num_samples, height, width, patch_size=(16, 256, 256),
                             num_patches_xy=3, num_patches_z=20):
    for sample_index in range(num_samples):
        slice_index = 0
        image_slices = []
        mask_slices = []
        current_image_dir = os.path.join(image_dir, str(sample_index))
        current_mask_dir = os.path.join(mask_dir, str(sample_index)) if mask_dir else None
        print(f"Processing {current_image_dir}")
        if not os.path.exists(current_image_dir):
            print(f"Directory does not exist: {current_image_dir}")
            continue
        if mask_dir and not os.path.exists(current_mask_dir):
            print(f"Directory does not exist: {current_mask_dir}")
            continue
        for file_name in os.listdir(current_image_dir):
            image_path = os.path.join(current_image_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            image_slices.append(image)
            if mask_dir:
                mask_path = os.path.join(current_mask_dir, file_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Failed to read mask: {mask_path}")
                    continue
                mask_slices.append(mask)
            slice_index += 1

        image_array = np.array(image_slices)
        image_array = np.reshape(image_array, (slice_index, height, width))
        if mask_dir:
            mask_array = np.array(mask_slices)
            mask_array = np.reshape(mask_array, (slice_index, height, width))
        else:
            mask_array = None

        save_image_and_mask_patches(image_array, mask_array, sample_index, patch_size=patch_size, num_patches_xy=num_patches_xy, num_patches_z=num_patches_z,
                                    image_output_dir=image_output_dir, mask_output_dir=mask_output_dir)


