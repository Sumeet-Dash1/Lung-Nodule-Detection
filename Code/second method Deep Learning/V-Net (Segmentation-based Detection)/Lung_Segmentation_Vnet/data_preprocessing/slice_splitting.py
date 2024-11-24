from __future__ import print_function, division
import os
import SimpleITK as sitk
import cv2
import numpy as np
from glob import glob

def get_image_depth_range(image):
    """
    Converts 3D Image array in 2D images along it's depth
    """
    start_flag = True
    start_position = 0
    end_position = 0
    for z in range(image.shape[0]):
        non_zero_flag = np.max(image[z])
        if non_zero_flag and start_flag:
            start_position = z
            start_flag = False
        if non_zero_flag:
            end_position = z
    return start_position, end_position

def resize_image(image, new_spacing, interpolation_method=sitk.sitkNearestNeighbor):
    """
   Resize the image using the SimpleITK ResampleImageFilter. 
   Provide the SimpleITK image, the new spacing values (e.g., [1, 1, 1]), 
   and the interpolation method (default: sitkNearestNeighbor). 
   The function will return the resampled image array and the resampled SimpleITK image.
    """
    new_spacing = np.array(new_spacing, float)
    original_spacing = image.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    original_size = image.GetSize()
    scaling_factor = new_spacing / original_spacing
    new_size = original_size / scaling_factor
    new_size = new_size.astype(np.int64)
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolation_method)
    resampled_image = resampler.Execute(image)
    if interpolation_method == sitk.sitkNearestNeighbor:
        resampled_image = sitk.Threshold(resampled_image, 0, 1.0, 255)
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    return resampled_array, resampled_image

def load_image(filename):
    """
    Load MHD files and normalize the intensity to the range 0-255 using the specified file path, returning a SimpleITK image.
    """
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.SetOutputMinimum(0)
    itk_image = rescale_filter.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itk_image

def load_image_with_truncation(filename, upper=200, lower=-200):
    """
    Load MHD files, truncate values outside a specified range, 
    and normalize to 0-255 using the specified file path, upper truncation value, 
    and lower truncation value, returning a SimpleITK image.
    """
    itk_image = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(itk_image)
    image_array[image_array > upper] = upper
    image_array[image_array < lower] = lower
    truncated_image = sitk.GetImageFromArray(image_array)
    truncated_image.SetSpacing(itk_image.GetSpacing())
    truncated_image.SetOrigin(itk_image.GetOrigin())
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.SetOutputMinimum(0)
    normalized_image = rescale_filter.Execute(sitk.Cast(truncated_image, sitk.sitkFloat32))
    return normalized_image

def process_training_data():
    slice_expansion = 13
    training_image_path = "preprocessed/image/"
    training_mask_path = "preprocessed/mask/"
    """
    Load ITK images, change z spacing to 1, and save image, lung mask, and tumor mask

    """
    series_index = 0
    for subset_index in range(10):
        base_path = "src/"
        subset_path = base_path + "subset" + str(subset_index) + "/"
        mask_output_path = "mask/"
        subset_mask_path = mask_output_path + "subset" + str(subset_index) + "/"
        image_files = glob(subset_path + "*.mhd")
        for file_index in range(len(image_files)):
            # Load ITK image and truncate values
            truncated_image = load_image_with_truncation(image_files[file_index], 600, -1000)
            image_filename = image_files[file_index][len(subset_path):-4]
            segmentation = sitk.ReadImage(subset_mask_path + image_filename + "_segmentation.mhd", sitk.sitkUInt8)
            z_spacing = segmentation.GetSpacing()[-1]
            # Change z spacing > 1.0 to 1.0
            if z_spacing > 1.0:
                _, segmentation = resize_image(segmentation, (segmentation.GetSpacing()[0], segmentation.GetSpacing()[1], 1.0),
                                               interpolation_method=sitk.sitkNearestNeighbor)
                _, truncated_image = resize_image(truncated_image, (truncated_image.GetSpacing()[0], truncated_image.GetSpacing()[1], 1.0),
                                                  interpolation_method=sitk.sitkLinear)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            truncated_image_array = sitk.GetArrayFromImage(truncated_image)

            image_output_path = training_image_path + str(series_index)
            mask_output_path = training_mask_path + str(series_index)
            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            if not os.path.exists(mask_output_path):
                os.makedirs(mask_output_path)
            # Get lung mask
            lung_mask = segmentation_array.copy()
            lung_mask[segmentation_array > 0] = 255
            # Get the ROI range of the mask, expand slices before and after, and get the expanded ROI image
            start_position, end_position = get_image_depth_range(lung_mask)
            if start_position == end_position:
                continue
            num_slices = np.shape(lung_mask)[0]
            start_position = start_position - slice_expansion
            end_position = end_position + slice_expansion
            if start_position < 0:
                start_position = 0
            if end_position > num_slices:
                end_position = num_slices
            truncated_image_array = truncated_image_array[start_position:end_position, :, :]
            lung_mask = lung_mask[start_position:end_position, :, :]
            # Save the source image, lung mask, and tumor mask
            for z in range(lung_mask.shape[0]):
                truncated_image_array = np.clip(truncated_image_array, 0, 255).astype('uint8')
                cv2.imwrite(os.path.join(image_output_path, f"{z}.bmp"), truncated_image_array[z])
                cv2.imwrite(os.path.join(mask_output_path, f"{z}.bmp"), lung_mask[z])
            series_index += 1

process_training_data()
