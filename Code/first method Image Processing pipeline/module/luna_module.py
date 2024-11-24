import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk

from scipy.ndimage import center_of_mass
from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation, binary_opening, binary_erosion
from skimage import measure, morphology
from skimage.segmentation import clear_border


"""Normalizations"""


def normalize_intensity(image, scale_min=0, scale_max=255, verbose=False):
    """
    Normalize the intensity values of an image to a specified scale.

    Args:
    image (numpy.ndarray): 2D array representing the input image.
    scale_min (int, optional): Minimum value of the desired scale range. Default is 0.
    scale_max (int, optional): Maximum value of the desired scale range. Default is 255.
    verbose (bool, optional): If True, print normalization details. Default is False.

    Returns:
    numpy.ndarray: Normalized image with values scaled to [scale_min, scale_max].

    Raises:
    ValueError: If the input is not a 2D array or if scale_min is not less than scale_max.
    """

    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Input image must be a 2D NumPy array.")

    if scale_min >= scale_max:
        raise ValueError("scale_min must be less than scale_max.")

    if len(np.unique(image)) < 2:
        if verbose:
            print(
                f"[NORMALIZE ({scale_min}, {scale_max})] - Only 1 intensity value -> setting to min scale value: {scale_min}")
        img = image.copy()
        img[:, :] = scale_min
        return img

    img_min = np.min(image)
    img_max = np.max(image)

    normalized_image = (image - img_min) / (img_max -
                                            img_min) * (scale_max - scale_min) + scale_min

    if verbose:
        print(
            f"[NORMALIZE ({scale_min}, {scale_max})] - Image min: {img_min}, max: {img_max}")

    return normalized_image


def norm2uint8(image):
    return normalize_intensity(image.copy(), 0, 255).astype(np.uint8)


def norm2float(image):
    return normalize_intensity(image.copy(), 0, 1).astype(float)


def norm2uint16(image):
    return normalize_intensity(image.copy(), 0, 65535).astype(np.uint16)


"""Operations with 3D images"""


def process_slices(img, axis_index, func, func_args=None):
    """
    Process all the slices of a 3D image along a specified axis using a given function.

    Parameters:
        image (numpy.ndarray): The input 3D image.
        axis_index (int): The index of the axis along which to iterate (0 for x, 1 for y, 2 for z).
        func (callable): The function to apply to each slice.
        func_args (dict): The arguments given to the 'func' parameter

    Returns:
        numpy.ndarray: The modified image.
    """
    image = img.copy()
    # Ensure axis_index is within valid range
    if axis_index < 0 or axis_index > 2:
        raise ValueError("Axis index must be 0, 1, or 2.")

    # Iterate over slices along the specified axis
    for idx in range(image.shape[axis_index]):
        # Extract the slice along the specified axis
        if axis_index == 0:
            slice_ = image[idx, :, :]
        elif axis_index == 1:
            slice_ = image[:, idx, :]
        else:
            slice_ = image[:, :, idx]

        # Apply the function to the slice
        modified_slice = func(slice_)

        # Update the slice in the original image
        if axis_index == 0:
            image[idx, :, :] = modified_slice
        elif axis_index == 1:
            image[:, idx, :] = modified_slice
        else:
            image[:, :, idx] = modified_slice
    return image


def get_slices(image, axis_index):
    """
    Extracts slices from a 3D image array along a specified axis.

    Args:
    image (numpy.ndarray): 3D array representing the input image.
    axis_index (int): Axis along which to extract slices (0, 1, or 2).

    Returns:
    numpy.ndarray: Array of 2D slices extracted along the specified axis.

    Raises:
    ValueError: If axis_index is not 0, 1, or 2.
    """
    slices = []
    # Ensure axis_index is within valid range
    if axis_index < 0 or axis_index > 2:
        raise ValueError("Axis index must be 0, 1, or 2.")

    # Iterate over slices along the specified axis
    for idx in range(image.shape[axis_index]):
        # Extract the slice along the specified axis
        if axis_index == 0:
            slice_ = image[idx, :, :]
        elif axis_index == 1:
            slice_ = image[:, idx, :]
        else:
            slice_ = image[:, :, idx]

        slices.append(slice_)

    return np.array(slices)


"""Plotting"""


def show_3_images(images, cmap="gray"):
    """
    Show 3 images in 1 row
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(images[0], cmap=cmap)
    axes[0].axis('off')  # Turn off axis ticks and labels
    axes[1].imshow(images[1], cmap=cmap)
    axes[1].axis('off')  # Turn off axis ticks and labels
    axes[2].imshow(images[2], cmap=cmap)
    axes[2].axis('off')  # Turn off axis ticks and labels

    return fig, axesF


def plot_slices(slices, cols=4, cmap="gray", titles=True):
    """
    Plots a series of image slices in a grid format.

    Args:
    slices (list of numpy.ndarray): List of 2D arrays representing image slices.
    cols (int, optional): Number of columns in the grid. Default is 4.
    cmap (str, optional): Colormap used to display the images. Default is "gray".
    titles (bool, optional): If True, display titles above each slice indicating its index. Default is True.

    Returns:
    None
    """
    rows = int(np.ceil(len(slices) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 4))

    # Flatten the array of axes for easy iteration
    axs = axs.ravel()

    for i, img in enumerate(slices):
        axs[i].imshow(img, cmap=cmap)
        if titles:
            axs[i].title.set_text(f"slice {i}")

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


def debugger(img, title=None):
    """
    Displays an image with its unique intensity values and an optional title.

    Args:
    img (numpy.ndarray): The image to be displayed, represented as a 2D array.
    title (str, optional): The title to be printed above the image. Default is None.

    Returns:
    None
    """
    if title is not None:
        print(title)
    print(np.unique(img))
    plt.imshow(img, cmap="gray")
    plt.axis('off')  # Turn off axis ticks and labels
    plt.show()


"""Image_dictionary search & helpers"""


def find_by_uid(uid, image_dict):
    dictionary = image_dict  # Search in 'image_dict'

    keys = list(dictionary.keys())
    for key in keys:
        if uid in dictionary[key]:
            return dictionary[key][uid]

    # If no uid is found None is returned
    print(f"NO {uid} UID")
    return None


def get_uids(image_dict):
    uids = []
    dictionary = image_dict  # Search in 'image_dict'
    keys = list(dictionary.keys())

    for key in keys:
        for uid in dictionary[key]:
            uids.append(uid)

    return uids


def subset_by_uid(uid, image_dict):
    dictionary = image_dict  # Search in 'image_dict'

    for subset in list(dictionary.keys()):
        for i, filename in enumerate(dictionary[subset]):
            if filename == uid:
                return subset, i

    print(f"NO {uid} UID")
    return None


def img_by_uid(uid, image_dict):
    img = find_by_uid(uid, image_dict)
    return sitk.GetArrayFromImage(img)


def meta_by_uid(uid, image_dict):
    img = find_by_uid(uid, image_dict)
    return img.GetOrigin(), img.GetSpacing()


def annotations_by_uid(uid, annotations_df):
    # Getting annotations for case
    mask = annotations_df['seriesuid'] == uid
    return annotations_df[mask]


def convert_annotation_df(annotations_df, verbose=False):
    """
    Converts an annotations DataFrame containing cartesian coordinates and diameters
    into a simplified DataFrame with separate columns for x, y, z coordinates and diameters.

    Parameters:
    annotations_df (pd.DataFrame): DataFrame with columns 'cartesian_coords(zyx)' and 'cartesian_diameters(zyx)'.
                                   Each entry in these columns is a string representation of coordinates or diameters.
    verbose (bool, optional): If True, prints the simplified DataFrame. Default is False.

    Returns:
    pd.DataFrame: A new DataFrame with columns 'x', 'y', 'z', 'diam_x', 'diam_y', 'diam_z' representing the
                  extracted coordinates and diameters.
    """

    # Extract and convert cartesian coordinates from string to numpy array
    node_coords = annotations_df["cartesian_coords(zyx)"].apply(
        lambda x: np.array(x.replace("(", "").replace(")",
                           "").split(",")).astype(int)[::-1]
    )
    node_coords = np.array(node_coords.to_list()).reshape(
        (len(node_coords.to_list()), 3))

    # Extract and convert cartesian diameters from string to numpy array
    diams = annotations_df["cartesian_diameters(zyx)"].apply(
        lambda x: np.array(x.replace("(", "").replace(")",
                           "").split(",")).astype(int)[::-1]
    )
    diams = np.array(diams.to_list()).reshape((len(diams.to_list()), 3))

    # Create a new DataFrame to store the extracted coordinates and diameters
    node_coords_df = pd.DataFrame(
        columns=["x", "y", "z", "diam_x", "diam_y", "diam_z"], index=annotations_df.index)
    node_coords_df["x"] = node_coords[:, 0]
    node_coords_df["y"] = node_coords[:, 1]
    node_coords_df["z"] = node_coords[:, 2]

    node_coords_df["diam_x"] = diams[:, 0]
    node_coords_df["diam_y"] = diams[:, 1]
    node_coords_df["diam_z"] = diams[:, 2]

    # If verbose is True, print the simplified DataFrame
    if verbose:
        print(f"[ANNOTATIONS] - simplified")
        print(node_coords_df)

    return node_coords_df


def convert_annotation_df_with_uid(annotations_df, verbose=False):
    """
    Converts an annotations DataFrame containing cartesian coordinates and diameters
    into a simplified DataFrame with separate columns for x, y, z coordinates and diameters.

    Parameters:
    annotations_df (pd.DataFrame): DataFrame with columns 'cartesian_coords(zyx)' and 'cartesian_diameters(zyx)'.
                                   Each entry in these columns is a string representation of coordinates or diameters.
    verbose (bool, optional): If True, prints the simplified DataFrame. Default is False.

    Returns:
    pd.DataFrame: A new DataFrame with columns 'x', 'y', 'z', 'diam_x', 'diam_y', 'diam_z' representing the
                  extracted coordinates and diameters.
    """

    # Extract and convert cartesian coordinates from string to numpy array
    node_coords = annotations_df["cartesian_coords(zyx)"].apply(
        lambda x: np.array(x.replace("(", "").replace(")",
                           "").split(",")).astype(int)[::-1]
    )
    node_coords = np.array(node_coords.to_list()).reshape(
        (len(node_coords.to_list()), 3))

    # Extract and convert cartesian diameters from string to numpy array
    diams = annotations_df["cartesian_diameters(zyx)"].apply(
        lambda x: np.array(x.replace("(", "").replace(")",
                           "").split(",")).astype(int)[::-1]
    )
    diams = np.array(diams.to_list()).reshape((len(diams.to_list()), 3))

    # Create a new DataFrame to store the extracted coordinates and diameters
    node_coords_df = pd.DataFrame(
        columns=["seriesuid", "x", "y", "z", "diam_x", "diam_y", "diam_z"], index=annotations_df.index)

    node_coords_df["seriesuid"] = annotations_df["seriesuid"]

    node_coords_df["x"] = node_coords[:, 0]
    node_coords_df["y"] = node_coords[:, 1]
    node_coords_df["z"] = node_coords[:, 2]

    node_coords_df["diam_x"] = diams[:, 0]
    node_coords_df["diam_y"] = diams[:, 1]
    node_coords_df["diam_z"] = diams[:, 2]

    # If verbose is True, print the simplified DataFrame
    if verbose:
        print(f"[ANNOTATIONS] - simplified")
        print(node_coords_df)

    return node_coords_df


"""Annotation drawing"""


def draw_ellipsoid(mask, center, diameters, zyx=True):
    """
    Draws ellipsioid with custom diameters on a 3D image
    NOTE: center coordinates are in (z,y,x) order by default
    """
    # Extract the diameters for each axis
    if zyx:
        dz, dy, dx = diameters
    else:
        dx, dy, dz = diameters

    # Calculate the radii for each axis
    rz, ry, rx = dz / 2, dy / 2, dx / 2

    # Extract the shape of the array
    zmax, ymax, xmax = mask.shape

    # Get the range of indices to iterate over (stay within the array bounds)
    z_start, z_end = max(0, center[0]-int(rz)), min(zmax, center[0]+int(rz)+1)
    y_start, y_end = max(0, center[1]-int(ry)), min(ymax, center[1]+int(ry)+1)
    x_start, x_end = max(0, center[2]-int(rx)), min(xmax, center[2]+int(rx)+1)

    # Iterate over each point in the bounding box of the ellipsoid
    for z in range(z_start, z_end):
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # Calculate the normalized distance from the current point to the center
                if (((x - center[2])**2 / rx**2) + ((y - center[1])**2 / ry**2) + ((z - center[0])**2 / rz**2)) <= 1:
                    mask[z, y, x] = 1  # Set the value inside the ellipsoid


def create_3d_mask(center, diameters, img):
    mask = np.zeros_like(img)
    draw_ellipsoid(mask, center, diameters)
    return mask


def create_annotations_mask(origin, spacing, annotations: pd.DataFrame, image: np.ndarray, verbose=False):
    """
    Creates a 3D mask for annotations based on provided metadata and CT scan.

    Parameters:
    origin (array): The origin coordinates of the CT scan.
    spacing (array): The spacing between the voxels in the CT scan.
    annotations (pd.DataFrame): DataFrame containing the annotations with columns 'coordX', 'coordY', 'coordZ', and 'diameter_mm'.
    image (np.ndarray): 3D numpy array representing the CT scan image.
    verbose (bool): If True, prints detailed debug information.

    Returns:
    mask (np.ndarray): 3D boolean mask where annotated regions are marked as True.
    indexes (np.ndarray): Array of annotation indexes.
    centers (np.ndarray): Array of annotation centers in voxel space.
    diameters (np.ndarray): Array of annotation diameters in voxel space.
    """
    centers = []
    diameters = []
    indexes = []

    # If there is no given annotations
    if not len(annotations):
        return np.zeros_like(image, dtype=np.uint8), np.array([]), np.array([]), np.array([]),

    # Iterate thorugh the annotations
    annotation_i = 0  # Toggle mask creation in the first iteration
    for i, row in annotations.iterrows():
        # Coordinates of the device
        world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])
        # Translated coordinates into voxelspace
        pixel_coords = np.round((world_coords - origin) / spacing).astype(int)

        diam = row["diameter_mm"]  # Diameter scalar
        # Diameters scaled by spacing (x, y, z)
        diams = np.rint(diam / np.array(spacing))

        # Flipping axes: (x, y, z) -> (z, y, x)
        # nodule center in voxel space (z,y,x ordering)
        v_center = np.array(
            [pixel_coords[2], pixel_coords[1], pixel_coords[0],])
        # flipping to (z, y, x)
        v_diameters = np.array([diams[2], diams[1], diams[0],])

        centers.append(v_center)
        diameters.append(v_diameters)
        indexes.append(i)

        # Adding annotations to the mask
        if annotation_i == 0:
            if verbose:
                print(f"[ANNOTATION #{annotation_i + 1}] -- Create image mask")

            mask = create_3d_mask(v_center, v_diameters, image)
            annotation_i += 1
        else:
            if verbose:
                print(
                    f"[ANNOTATION #{annotation_i + 1}] -- Adding annotation #{annotation_i + 1}, index: {i}")

            draw_ellipsoid(mask, v_center, v_diameters)
            annotation_i += 1

        if verbose:
            z, y, x = image.shape
            print(f"Row index: {i}")
            print(f"Slices: Z: {z}, Y: {y}, X: {x}")
            print(f"Origin : {origin}")
            print(f"Spacing : {spacing}")
            print(f"V-center: {v_center}")
            print(f"Diameter: {diam}")
            print(f"V-diameters: {v_diameters}")

    return mask.astype(bool), np.array(indexes), np.array(centers), np.array(diameters)


def create_annotations_mask_by_uid(uid: str, image_dict: dict, annotations_df: pd.DataFrame, verbose=False):
    """
    Creates a 3D mask for annotations based on the provided UID and metadata.

    Parameters:
    uid (str): Unique identifier for the CT scan case.
    image_dict (dict): Dictionary containing image metadata.
    annotations_df (pd.DataFrame): DataFrame containing the annotations.
    verbose (bool): If True, prints detailed debug information.

    Returns:
    mask (np.ndarray): 3D boolean mask where annotated regions are marked as True.
    indexes (np.ndarray): Array of annotation indexes.
    centers (np.ndarray): Array of annotation centers in voxel space.
    diameters (np.ndarray): Array of annotation diameters in voxel space.
    """
    origin, spacing = meta_by_uid(uid, image_dict)
    annotations = annotations_by_uid(uid, annotations_df)
    img = img_by_uid(uid, image_dict)

    return create_annotations_mask(origin, spacing, annotations, img, verbose)


def masked_annotations_by_uid(uid, image_dict, annotations_df, verbose=False):
    """
    Gets masked annotations based on the provided UID and metadata.

    Parameters:
    uid (str): Unique identifier for the CT scan case.
    image_dict (dict): Dictionary containing image metadata.
    annotations_df (pd.DataFrame): DataFrame containing the annotations.
    verbose (bool): If True, prints detailed debug information.

    Returns:
    masked_annotations (np.ndarray): Array of 2D masked annotations.
    """
    img_3d = img_by_uid(uid, image_dict)

    mask_3d, indexes, centers, diameters = create_annotations_mask_by_uid(
        uid, image_dict, annotations_df)
    center_z_slices = centers[:, 0]

    mask_slices = mask_3d[center_z_slices, :, :]
    img_slices = img_3d[center_z_slices, :, :]

    if verbose:
        pairs = []
    masked_annotations = []

    for i in range(len(center_z_slices)):

        masked = img_slices[i].copy()
        masked[~mask_slices[i]] = img_slices[i].min()

        masked_annotations.append(masked)

        if verbose:
            pairs.append(img_slices[i])
            pairs.append(mask_slices[i])
            pairs.append(masked)
            plot_slices(pairs, 3)
            print(f"Pixel intensities on masked image")
            print(np.unique(masked))

    return np.array(masked_annotations)


def masked_annotations_with_info_by_uid(uid, image_dict, annotations_df, verbose=False):
    """
    Gets masked annotations with additional information based on the provided UID and metadata.

    Parameters:
    uid (str): Unique identifier for the CT scan case.
    image_dict (dict): Dictionary containing image metadata.
    annotations_df (pd.DataFrame): DataFrame containing the annotations.
    verbose (bool): If True, prints detailed debug information.

    Returns:
    masked_annotations (np.ndarray): Array of 2D masked annotations.
    indexes (np.ndarray): Array of annotation indexes.
    centers (np.ndarray): Array of annotation centers in voxel space.
    diameters (np.ndarray): Array of annotation diameters in voxel space.
    """
    img_3d = img_by_uid(uid, image_dict)

    mask_3d, indexes, centers, diameters = create_annotations_mask_by_uid(
        uid, image_dict, annotations_df)
    center_z_slices = centers[:, 0]

    mask_slices = mask_3d[center_z_slices, :, :]
    img_slices = img_3d[center_z_slices, :, :]

    if verbose:
        pairs = []
    masked_annotations = []

    for i in range(len(center_z_slices)):

        masked = img_slices[i].copy()
        masked[~mask_slices[i]] = img_slices[i].min()

        masked_annotations.append(masked)

        if verbose:
            pairs.append(img_slices[i])
            pairs.append(mask_slices[i])
            pairs.append(masked)
            print(f"Pixel intensities on masked image")
            print(np.unique(masked))

    if verbose:
        plot_slices(pairs, 3)

    return np.array(masked_annotations), indexes, centers, diameters


"""Lung segmentation"""


AIR_TH = -1024
LUNG_TH = -500


def remove_non_central_objects(img, debug=False):
    """
    Deletes objects from the input binary image slice that are not within the central region.

    Parameters:
    img (ndarray): Input binary image slice.
    debug (bool): If True, shows debugging information (default is False).

    Returns:
    filtered_img (ndarray): Modified binary image slice with certain objects removed.
    """
    # Create a copy of the input slice to
    filtered_img = img.copy()

    # Label connected components in the slice
    labels = measure.label(img, background=0)
    idxs = np.unique(labels)[1:]  # Get unique labels, excluding the background

    # Compute the center of mass for each label
    COM_ys = np.array([center_of_mass(labels == i)[0] for i in idxs])
    masses = [center_of_mass(labels == j)
              for j in range(1, np.unique(labels).size)]

    if debug:
        # Display debugging information if debug is True
        plt.imshow(filtered_img)
        plt.show()
        plt.imshow(labels)
        plt.show()
        print(masses)

    # Calculate distances from the center of the slice
    center_y = img.shape[0] / 2
    distances = [(i, np.abs(com[0] - center_y))
                 for i, com in zip(idxs, masses)]

    # Sort by distance and keep the closest 3 labels
    distances.sort(key=lambda x: x[1])
    closest_idxs = [distances[i][0] for i in range(min(3, len(distances)))]

    # Remove labels that are not among the closest 3
    for idx in idxs:
        if idx not in closest_idxs:
            filtered_img[labels == idx] = 0

    # Further filter labels based on their vertical position in the slice
    for idx, COM_y in zip(idxs, COM_ys):
        if COM_y < 0.25 * img.shape[0] or COM_y > 0.75 * img.shape[0]:
            filtered_img[labels == idx] = 0

    return filtered_img


def unwanted_object_filter(img: np.ndarray, area_th=1500) -> np.ndarray:
    """
    Filters out unwanted objects from a binary image based on area and location.

    Parameters:
    img (ndarray): Input binary image.

    Returns:
    filtered_image (ndarray): Binary image with unwanted objects removed.
    """
    # Perform binary closing to close small gaps in the image
    closed = binary_closing(img.copy(), morphology.disk(5))

    # Label connected components in the closed image
    labels = measure.label(closed)
    properties = measure.regionprops(labels)

    # Sort regions by area and keep those larger than a threshold
    detected_objects = [obj for obj in properties if obj.area > area_th]

    # Create an empty image to hold the result
    filtered_image = np.zeros_like(img, dtype=bool)

    # Fill in the regions of the relevant objects
    for prop in detected_objects:
        filtered_image[labels == prop.label] = True

    # Remove tables or similar objects using the delete_table function
    filtered_image = remove_non_central_objects(filtered_image)

    return filtered_image


def binarize_lung(img: np.ndarray, threshold_val=LUNG_TH):
    """
    Binarizes a lung image using a series of image processing steps.

    This function applies Gaussian blur, thresholding, and several morphological operations 
    to binarize the lung image, removing unwanted artifacts and filling holes.

    Args:
    img (numpy.ndarray): The input lung image.
    threshold_val (int or float): The threshold value for binarization.

    Returns:
    numpy.ndarray: The binarized lung image.
    """
    # Apply Gaussian blur to the image
    blurred = cv.GaussianBlur(img, (5, 5), 0)

    # Apply thresholding to create a binary image
    th = (blurred < threshold_val).astype(float)

    # Remove objects connected to the image border
    th = clear_border(th)

    # Remove small unwanted objects
    th = unwanted_object_filter(th)

    # Apply binary closing to smooth edges and remove small holes
    th = binary_closing(th, morphology.disk(3))

    # Fill remaining larger holes
    th = binary_fill_holes(th)

    return th


def binarize_lung_3d(img_3d, threshold_val=LUNG_TH):
    """
    Binarizes a 3D lung image using a specified threshold value.

    Args:
    img_3d (numpy.ndarray): The input 3D lung image.
    threshold_val (int): The threshold value for binarization.

    Returns:
    numpy.ndarray: The binarized 3D lung image.

    Raises:
    Exception: If the input image is not 3-dimensional.
    """
    if len(img_3d.shape) != 3:
        raise Exception("Shape has to be 3 dimensional")
    return process_slices(img_3d, 0, binarize_lung)


"""Nodule extraction"""


def get_slice_candidates_old(img, z, nthreshold=-400, pthreshold=200, debug=False, debug_res=False):
    # Get the minimum value of the image
    min_val = AIR_TH

    # Create a copy of the image
    img_n = img.copy()
    if debug:
        debugger(img_n, "img_n")

    # Apply thresholding to the image
    # img_n = np.where(img < nthreshold, min_val, img_n)
    img_n[img < nthreshold] = AIR_TH
    # img_n = np.where(img < nthreshold, min_val, img_n)
    img_n[img > pthreshold] = AIR_TH
    if debug:
        debugger(img_n, "img_n")

    # Normalize the image
    inp = norm2uint16(img_n)
    if debug:
        debugger(inp, "inp")

    # Apply Gaussian blur to the normalized image
    blurred = cv.GaussianBlur(inp, (5, 5), 0)
    if debug:
        debugger(blurred, "blurred")

    # Convert the image to 8-bit
    image_8bit = norm2uint8(blurred)
    if debug:
        debugger(image_8bit, "image_8bit")

    # Apply Otsu's thresholding
    ret, otsu_img = cv.threshold(
        image_8bit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_img = norm2float(otsu_img)
    if debug:
        debugger(otsu_img, "otsu_img")

    # Binarize the original image
    lung_mask = binarize_lung(img).astype(float)
    if debug:
        debugger(lung_mask, "lung_mask")

    # Apply morphological closing to the lung mask
    lung_mask = cv.morphologyEx(
        lung_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35)))
    if debug:
        debugger(lung_mask, "lung_mask_closed")

    # Apply morphological erosion to the thresholded image
    kernel_size = 5
    eroded = cv.morphologyEx(otsu_img, cv.MORPH_ERODE,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    if debug:
        debugger(eroded, "eroded")

    # Apply morphological closing with a larger kernel
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing_img = cv.morphologyEx(otsu_img, cv.MORPH_CLOSE, kernel)
    if debug:
        debugger(closing_img, "closing_img")

    # Combine the lung mask and the closed image to get the region of interest
    roi = lung_mask * closing_img
    if debug:
        debugger(roi, "roi")

    # Invert the lung mask
    inverted_image = (~lung_mask.astype(bool)).astype(float)
    if debug:
        debugger(inverted_image, "inverted_image")

    # Bitwise AND between the inverted lung mask and the closed image
    result_image = inverted_image * closing_img
    if debug:
        debugger(result_image, "result_image")

    # Apply morphological opening with a larger kernel to the closed image
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    marker = cv.morphologyEx(closing_img.astype(float), cv.MORPH_OPEN, kernel1)
    if debug:
        debugger(marker, "marker")

    # Morphological reconstruction by iterative dilations
    mask_geo = result_image.copy()
    iterations = 0

    while True:
        marker_prev = marker.copy()
        marker = cv.dilate(
            marker, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        marker = np.minimum(marker, mask_geo)  # Geodesic dilation

        if np.array_equal(marker, marker_prev):
            break
    if debug:
        debugger(mask_geo, "mask_geo")

    # Display the difference (e.g., stars in the galaxy)
    difference = cv.absdiff(closing_img, marker)
    if debug:
        debugger(difference, "difference")

    # Update the region of interest
    roi_new = lung_mask * difference
    if debug:
        debugger(roi_new, "roi_new")

    # Apply morphological opening to the updated region of interest
    roi_new = cv.morphologyEx(
        roi_new, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    if debug:
        debugger(roi_new, "roi_new_opened")

    # Compute the distance transform
    distance_transform = cv.distanceTransform(
        norm2uint8(roi_new), cv.DIST_L1, 3)
    if debug:
        debugger(distance_transform, "distance_transform")

    distance_transform_vis = norm2uint8(distance_transform)
    if debug:
        debugger(distance_transform_vis, "distance_transform_vis")

    # Apply morphological opening to the distance transform
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded_distance_transform = cv.morphologyEx(
        distance_transform_vis, cv.MORPH_OPEN, kernel2)
    if debug:
        debugger(eroded_distance_transform, "eroded_distance_transform")

    # Threshold the eroded distance transform to get internal markers

    _, internal_markers = cv.threshold(
        eroded_distance_transform, 0, 127, cv.THRESH_BINARY)  # NOTE: HARDCODED TH!!!
    if debug:
        debugger(internal_markers, "internal_markers")

    # Find contours in the internal markers
    contours, _ = cv.findContours(
        internal_markers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    conts = []

    # Compute centroids of the contours and keep the ones that fit the expectations (area < 1300)
    cx_list = []
    cy_list = []
    for contour in contours:
        M = cv.moments(contour)
        if cv.contourArea(contour) < 1300 and M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cx_list.append(cx)
            cy_list.append(cy)
            conts.append(contour)

    # DataFrame with the centroids
    coord = pd.DataFrame({'x': cx_list, 'y': cy_list, 'z': z})

    candidate_mask = np.zeros_like(img, dtype=float)
    cv.drawContours(candidate_mask, conts, -1, 1, -1)
    if debug:
        debugger(candidate_mask, "candidate_mask")

    if debug_res:
        show_3_images([inp, lung_mask, candidate_mask])

    return coord, candidate_mask


def get_slice_candidates(img, z, nthreshold=-400, pthreshold=200, debug=False, debug_res=False):
    # Get the minimum value of the image
    min_val = AIR_TH

    # Create a copy of the image
    img_n = img.copy()
    if debug:
        debugger(img_n, "img_n")

    # Apply thresholding to the image
    # img_n = np.where(img < nthreshold, min_val, img_n)
    img_n[img < nthreshold] = AIR_TH
    # img_n = np.where(img < nthreshold, min_val, img_n)
    img_n[img > pthreshold] = AIR_TH
    if debug:
        debugger(img_n, "img_n")

    # Normalize the image
    inp = norm2uint16(img_n)
    if debug:
        debugger(inp, "inp")

    # Apply Gaussian blur to the normalized image
    blurred = cv.GaussianBlur(inp, (5, 5), 0)
    if debug:
        debugger(blurred, "blurred")

    # Convert the image to 8-bit
    image_8bit = norm2uint8(blurred)
    if debug:
        debugger(image_8bit, "image_8bit")

    # Apply Otsu's thresholding
    ret, otsu_img = cv.threshold(
        image_8bit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_img = norm2float(otsu_img)
    if debug:
        debugger(otsu_img, "otsu_img")

    # Binarize the original image
    lung_mask = binarize_lung(img).astype(float)
    if debug:
        debugger(lung_mask, "lung_mask")

    # Apply morphological closing to the lung mask
    lung_mask = cv.morphologyEx(
        lung_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35)))
    if debug:
        debugger(lung_mask, "lung_mask_closed")

    # Apply morphological erosion to the thresholded image
    kernel_size = 5
    eroded = cv.morphologyEx(otsu_img, cv.MORPH_ERODE,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    if debug:
        debugger(eroded, "eroded")

    # Apply morphological closing with a larger kernel
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing_img = cv.morphologyEx(otsu_img, cv.MORPH_CLOSE, kernel)
    if debug:
        debugger(closing_img, "closing_img")

    # Combine the lung mask and the closed image to get the region of interest
    roi = lung_mask * closing_img
    if debug:
        debugger(roi, "roi")

    # Apply morphological opening to the updated region of interest
    roi_new = cv.morphologyEx(
        roi, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    if debug:
        debugger(roi_new, "roi_new_opened")

    # Compute the distance transform
    distance_transform = cv.distanceTransform(
        norm2uint8(roi_new), cv.DIST_L1, 3)
    if debug:
        debugger(distance_transform, "distance_transform")

    distance_transform_vis = norm2uint8(distance_transform)
    if debug:
        debugger(distance_transform_vis, "distance_transform_vis")

    # Apply morphological opening to the distance transform
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded_distance_transform = cv.morphologyEx(
        distance_transform_vis, cv.MORPH_OPEN, kernel2)
    if debug:
        debugger(eroded_distance_transform, "eroded_distance_transform")

    # Threshold the eroded distance transform to get internal markers

    _, internal_markers = cv.threshold(
        eroded_distance_transform, 0, 127, cv.THRESH_BINARY)  # NOTE: HARDCODED TH!!!
    if debug:
        debugger(internal_markers, "internal_markers")

    # Find contours in the internal markers
    contours, _ = cv.findContours(
        internal_markers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Label connected components in the closed image
    labels = measure.label(internal_markers)
    properties = measure.regionprops(labels)

    if debug:
        debugger(labels)

    if debug:
        plt.imshow(labels)
        plt.show()

    # Sort regions by area and keep those larger than a threshold
    detected_objects = [
        obj for obj in properties if obj.area < 1500]  # area filtering
    # eccentricity filtering
    detected_objects = [obj for obj in properties if obj.eccentricity < 0.7500]
    # solidity filtering
    detected_objects = [obj for obj in properties if obj.solidity > 0.600]

    # Compute centroids of the contours and keep the ones that fit the expectations (area < 1300)
    cx = []
    cy = []
    candidate_mask = np.zeros_like(img, dtype=float)
    for obj in detected_objects:
        candidate_mask[labels == obj.label] = 1.
        if debug_res:
            debugger(candidate_mask)
        centroid = obj.centroid
        cy.append(centroid[0])  # y-coordinate
        cx.append(centroid[1])

    # DataFrame with the centroids
    coord = pd.DataFrame({'x': np.round(cx).astype(
        int), 'y': np.round(cy).astype(int), 'z': z})

    if debug:
        debugger(candidate_mask, "candidate_mask")

    if debug_res:
        show_3_images([inp, lung_mask, candidate_mask])

    return coord, candidate_mask

def filter_slice_on_y_axis(y_slice, area_th=(48*48),  debug=False):
  if not len(y_slice.nonzero()[0]):
    if debug:
      print(f"[SKIP] - empty mask at slice ")
      return y_slice.astype(bool)
  if debug:
    plt.imshow(y_slice)
    plt.show()

  # Label connected components in the closed image
  labels = measure.label(y_slice)
  properties = measure.regionprops(labels)

  if debug:
    plt.imshow(labels)
    plt.show()

  # Sort regions by area and keep those larger than a threshold
  detected_objects = [obj for obj in properties if obj.area < area_th and obj.area] # area filtering
  detected_objects = [obj for obj in detected_objects if obj.solidity > 0.65] # solidity filtering

  # Create an empty image to hold the result
  filtered_image = np.zeros_like(y_slice, dtype=bool)

  # Fill in the regions of the relevant objects
  for prop in detected_objects:
      if debug:
        print(f"label: {prop.label}")
        print(f"area: {prop.area}")
        print(f"eccentricity: {prop.eccentricity}")
        print(f"solidity: {prop.solidity}")
      filtered_image[labels == prop.label] = True

  if debug:
    plt.imshow(filtered_image)
    plt.show()

  return filtered_image


def process_slice_candidates_2_axes(img_3d, verbose=False, debug=False):
  # Candidate extraction from Z axis
  z_centers_df, masks = process_slice_candidates_z(img_3d, )

  # Reconstruction of 3d mask
  mask_img_3d = np.array(masks)

  # Filtering out false candidates on Y axis
  y_filtered_mask_img_3d = process_slices(mask_img_3d, axis_index=1, func=filter_slice_on_y_axis,)

  # Identify connected components in 3D mask
  labeled_mask = measure.label(y_filtered_mask_img_3d, connectivity=1)

  num_features = labeled_mask.max()

  # Calculate the centers of mass for each object
  centers = center_of_mass(y_filtered_mask_img_3d, labeled_mask, range(1, num_features + 1))
  centers = np.array(centers)

  y_centers_df = pd.DataFrame({
      "x": np.round(centers[:,2]).astype(int),
      "y": np.round(centers[:,1]).astype(int),
      "z": np.round(centers[:,0]).astype(int),
  })
  return y_centers_df, z_centers_df



def process_slice_candidates(img, verbose=False, debug=False):
    """
    Process each slice of a 3D image to identify candidate regions of interest.

    Parameters:
    img (numpy.ndarray): A 3D image represented as a numpy array.
    verbose (bool): If True, print detailed progress information. Default is False.

    Returns:
    pd.DataFrame: A dataframe containing candidate coordinates (x, y, z).
    list: A list of masks corresponding to the candidate regions in each slice.
    """
    # Initialize an empty DataFrame to store candidate coordinates
    candidates = pd.DataFrame(columns=["x", "y", "z"])
    # Initialize an empty list to store candidate masks for each slice
    candidates_masks = []

    # Iterate over slices along the first axis (z-axis)
    for slice_i in range(img.shape[0]):
        if verbose:
            print(f"[START] - processing slice #{slice_i}")

        # Extract the current slice along the z-axis
        slice_ = img[slice_i, :, :]

        # Apply the function to identify candidates and their masks in the current slice
        slice_candidates, slice_candidates_mask = get_slice_candidates(
            slice_, slice_i, debug_res=debug)

        if verbose:
            print(
                f"[DONE] - {len(slice_candidates)} candidates found for slice #{slice_i}")

        # Append the found candidates to the main candidates DataFrame
        candidates = pd.concat(
            [candidates, slice_candidates], ignore_index=True)

        # Add the mask for the current slice to the candidates_masks list
        candidates_masks.append(slice_candidates_mask)

        if verbose:
            print(f"\n[STATUS] - {len(candidates)} candidates found\n" +
                  f"[STATUS] - {img.shape[0]-slice_i} slices left\n")

    return candidates, candidates_masks


"""Feature extraction"""


def create_patch(img, patch_center, patch_size=48):
    """
    Creates a patch from the input image centered at the given coordinates.
    If the patch extends beyond the image boundaries, the out-of-bound areas
    are filled with the air threshold value.

    Parameters:
    img (ndarray): Input image.
    patch_center (tuple): Coordinates (x, y) of the patch center.
    patch_size (int): Size of the patch (default is 48).

    Returns:
    patch (ndarray): The extracted patch from the image, with out-of-bound areas filled.
    coords (tuple): Coordinates of the top-left and bottom-right corners of the patch.
    """
    im = img.copy()  # Make a copy of the input image to avoid modifying the original
    x, y = patch_center
    from_center = patch_size / 2

    # Calculate top-left and bottom-right coordinates of the patch
    tl_y, tl_x = int(y - from_center), int(x - from_center)
    br_y, br_x = int(y + from_center), int(x + from_center)

    # Pad the image if necessary
    pad_y = max(0, -tl_y, br_y - im.shape[0])
    pad_x = max(0, -tl_x, br_x - im.shape[1])

    if pad_y > 0 or pad_x > 0:
        im = np.pad(im, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode='constant', constant_values=AIR_TH)
        tl_y += pad_y
        tl_x += pad_x
        br_y += pad_y
        br_x += pad_x

    # Extract the patch from the image
    patch = im[tl_y:br_y, tl_x:br_x]

    return patch, ((tl_x, tl_y), (br_x, br_y))


"""VALIDATION"""


def find_neighborhood_indices(large_df, small_df):
    """
    Find indices in the larger DataFrame where the coordinates are within a
    specified neighborhood distance from any row coordinates in the smaller DataFrame.

    Parameters:
    large_df (pd.DataFrame): The larger DataFrame with 'x', 'y', 'z','diam_x', 'diam_y', 'diam_z' columns.
    small_df (pd.DataFrame): The smaller DataFrame with 'x', 'y', 'z','diam_x', 'diam_y', 'diam_z' columns.

    Returns:
    np.ndarray: An array of indices from the larger DataFrame that are within the neighborhood.
    """
    # Initialize an empty list to store the indices
    neighborhood_indices = []
    neighborhood_indices_dict = {}

    # Iterate through each row in the smaller DataFrame
    for i, row in small_df.iterrows():
        neighborhood_indices_dict[i] = []
        # Create conditions to check for vicinity in all three axes
        a_x = large_df["x"] >= (row["x"] - row["diam_x"])
        b_x = large_df["x"] <= (row["x"] + row["diam_x"])
        a_y = large_df["y"] >= (row["y"] - row["diam_y"])
        b_y = large_df["y"] <= (row["y"] + row["diam_y"])
        a_z = large_df["z"] >= (row["z"] - row["diam_z"])
        b_z = large_df["z"] <= (row["z"] + row["diam_z"])

        # Combine conditions for all three axes
        vicinity_condition = a_x & b_x & a_y & b_y & a_z & b_z

        # Find indices where all conditions are satisfied
        close_indices = large_df.index[vicinity_condition].tolist()

        # Append these indices to the neighborhood_indices list
        neighborhood_indices.extend(close_indices)
        neighborhood_indices_dict[i] = (close_indices)

    # Return unique indices as a numpy array
    return np.unique(neighborhood_indices), neighborhood_indices_dict


def sensitivity_score(candidates_df, annotations_df, threshold=15):
    """
    Calculate the sensitivity score based on candidate detections and annotated coordinates.

    Parameters:
    candidates_df (pd.DataFrame): DataFrame containing candidate coordinates with 'x', 'y', 'z' ,'diam_x', 'diam_y', 'diam_z'columns.
    annotations_df (pd.DataFrame): DataFrame containing annotated coordinates in the format with 'x', 'y', 'z', 'diam_x', 'diam_y', 'diam_z' columns.

    Returns:
    float: Sensitivity score, calculated as the ratio of correctly identified annotations to the total number of annotations.
    """

    # Find neighborhood indices within a threshold distance
    candidate_indices, candidate_dict = find_neighborhood_indices(
        candidates_df, annotations_df)

    # Initialize a score counter
    score = 0

    # Iterate over the candidate dictionary to calculate the score
    for key, val in candidate_dict.items():
        # If there is a candidate in the vicinity for the annotation, it counts as a 'hit'
        if len(candidate_dict[key]):
            score += 1

    # Return the sensitivity score
    return score / len(annotations_df)


def find_neighborhood_indices_more_precise(large_df, small_df):
    """
    Find indices in the larger DataFrame where the coordinates are within a
    specified neighborhood distance from any row coordinates in the smaller DataFrame.

    Parameters:
    large_df (pd.DataFrame): The larger DataFrame with 'x', 'y', 'z','diam_x', 'diam_y', 'diam_z' columns.
    small_df (pd.DataFrame): The smaller DataFrame with 'x', 'y', 'z','diam_x', 'diam_y', 'diam_z' columns.

    Returns:
    np.ndarray: An array of indices from the larger DataFrame that are within the neighborhood.
    """
    # Initialize an empty list to store the indices
    neighborhood_indices = []
    neighborhood_indices_dict = {}

    # Iterate through each row in the smaller DataFrame
    for i, row in small_df.iterrows():
        neighborhood_indices_dict[i] = []
        # Create conditions to check for vicinity in all three axes
        a_x = large_df["x"] >= (row["x"] - ((row["diam_x"] / 2) + 2))
        b_x = large_df["x"] <= (row["x"] + ((row["diam_x"] / 2) + 2))
        a_y = large_df["y"] >= (row["y"] - ((row["diam_y"] / 2) + 2))
        b_y = large_df["y"] <= (row["y"] + ((row["diam_y"] / 2) + 2))
        a_z = large_df["z"] >= (row["z"] - ((row["diam_z"] / 2) + 2))
        b_z = large_df["z"] <= (row["z"] + ((row["diam_z"] / 2) + 2))

        # Combine conditions for all three axes
        vicinity_condition = a_x & b_x & a_y & b_y & a_z & b_z

        # Find indices where all conditions are satisfied
        close_indices = large_df.index[vicinity_condition].tolist()

        # Append these indices to the neighborhood_indices list
        neighborhood_indices.extend(close_indices)
        neighborhood_indices_dict[i] = (close_indices)

    # Return unique indices as a numpy array
    return np.unique(neighborhood_indices), neighborhood_indices_dict


def sensitivity_score_more_precise(candidates_df, annotations_df, threshold=15):
    """
    Calculate the sensitivity score based on candidate detections and annotated coordinates.

    Parameters:
    candidates_df (pd.DataFrame): DataFrame containing candidate coordinates with 'x', 'y', 'z' ,'diam_x', 'diam_y', 'diam_z'columns.
    annotations_df (pd.DataFrame): DataFrame containing annotated coordinates in the format with 'x', 'y', 'z', 'diam_x', 'diam_y', 'diam_z' columns.

    Returns:
    float: Sensitivity score, calculated as the ratio of correctly identified annotations to the total number of annotations.
    """

    # Find neighborhood indices within a threshold distance
    candidate_indices, candidate_dict = find_neighborhood_indices_more_precise(
        candidates_df, annotations_df)

    # Initialize a score counter
    score = 0

    # Iterate over the candidate dictionary to calculate the score
    for key, val in candidate_dict.items():
        # If there is a candidate in the vicinity for the annotation, it counts as a 'hit'
        if len(candidate_dict[key]):
            score += 1

    # Return the sensitivity score
    return score / len(annotations_df)
