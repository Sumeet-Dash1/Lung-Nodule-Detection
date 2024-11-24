from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd

tqdm = lambda x: x

def create_nodule_mask(mask, voxel_center, diameter_mm, voxel_spacing):
    diameter_voxels_z = int(diameter_mm / voxel_spacing[2] + 1)
    diameter_voxels_y = int(diameter_mm / voxel_spacing[1] + 1)
    diameter_voxels_x = int(diameter_mm / voxel_spacing[0] + 1)
    diameter_voxels_z = np.rint(diameter_voxels_z / 2)
    diameter_voxels_y = np.rint(diameter_voxels_y / 2)
    diameter_voxels_x = np.rint(diameter_voxels_x / 2)
    z_min = int(voxel_center[0] - diameter_voxels_z)
    z_max = int(voxel_center[0] + diameter_voxels_z + 1)
    x_min = int(voxel_center[1] - diameter_voxels_x)
    x_max = int(voxel_center[1] + diameter_voxels_x + 1)
    y_min = int(voxel_center[2] - diameter_voxels_y)
    y_max = int(voxel_center[2] + diameter_voxels_y + 1)
    mask[z_min:z_max, x_min:x_max, y_min:y_max] = 1.0
    print((z_max - z_min, x_max - x_min, y_max - y_min))

def find_image_file_path(file_list, case_id):
    for file_path in file_list:
        if case_id in file_path:
            return file_path

# Processing image files and saving mask image files
for subset_index in range(1):
    base_path = "src/"
    subset_path = base_path + "subset" + str(subset_index) + "/"
    mask_output_path = "mask/"
    subset_mask_path = mask_output_path + "subset" + str(subset_index) + "/"
    if not os.path.exists(subset_mask_path):
        os.makedirs(subset_mask_path)
    image_file_list = glob(subset_path + "*.mhd")
    
    image_file_paths = [file_path[:-4] for file_path in image_file_list]

    # Load nodule locations
    annotations_csv_path = "E:/Python_programs/Lung_Segmentation_Vnet/"
    nodule_annotations_df = pd.read_csv(annotations_csv_path + "annotations.csv")
    nodule_annotations_df["file"] = nodule_annotations_df["seriesuid"].map(lambda file_name: find_image_file_path(image_file_paths, file_name))
    nodule_annotations_df = nodule_annotations_df.dropna()
    
    # Loop over image files
    for file_index, image_file_path in enumerate(tqdm(image_file_paths)):
        nodules_in_file_df = nodule_annotations_df[nodule_annotations_df["file"] == image_file_path]
        full_image_file_path = image_file_path + ".mhd"
        itk_image = sitk.ReadImage(full_image_file_path)
        image_array = sitk.GetArrayFromImage(itk_image)
        num_slices, image_height, image_width = image_array.shape
        image_origin = np.array(itk_image.GetOrigin())
        image_spacing = np.array(itk_image.GetSpacing())
        
        nodule_mask_array = np.zeros((num_slices, image_height, image_width), dtype=np.float64)
        
        if not nodules_in_file_df.empty:
            for _, nodule in nodules_in_file_df.iterrows():
                nodule_center_world = np.array([nodule["coordX"], nodule["coordY"], nodule["coordZ"]])
                nodule_diameter_mm = nodule["diameter_mm"]
                voxel_center = np.rint((nodule_center_world - image_origin) / image_spacing)
                voxel_center[0], voxel_center[1], voxel_center[2] = voxel_center[2], voxel_center[1], voxel_center[0]
                create_nodule_mask(nodule_mask_array, voxel_center, nodule_diameter_mm, image_spacing)

        nodule_mask_array = np.uint8(nodule_mask_array * 255.)
        nodule_mask_array = np.clip(nodule_mask_array, 0, 255).astype('uint8')
        mask_image = sitk.GetImageFromArray(nodule_mask_array)
        mask_image.SetSpacing(image_spacing)
        mask_image.SetOrigin(image_origin)
        output_file_name = full_image_file_path[len(subset_path):-4] + "_segmentation.mhd"
        sitk.WriteImage(mask_image, subset_mask_path + output_file_name)
