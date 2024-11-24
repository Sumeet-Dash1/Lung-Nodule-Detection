import os
import cv2
import numpy as np
import pandas as pd
from Vnet.model_vnet3d import Vnet3dModule
from Vnet.layer import save_images
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.src.legacy.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce_loss = K.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce_loss + d_loss

def compute_iou(y_true, y_pred, threshold=0.5):
    y_pred_thresh = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_thresh)
    union = np.sum(y_true) + np.sum(y_pred_thresh) - intersection
    if union == 0:
        return 1.0
    else:
        return intersection / union

def compute_sensitivity(y_true, y_pred, threshold=0.5):
    y_pred_thresh = (y_pred > threshold).astype(np.float32)
    tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
    fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
    if (tp + fn) == 0:
        return 1.0
    else:
        return tp / (tp + fn)

def predict():
    base_src_path = "segmentation/Image/"
    base_mask_path = "segmentation/Mask/"
    folders = ["310_69"]
    results = []

    for folder in folders:
        src_path = os.path.join(base_src_path, folder) + "/"
        mask_path = os.path.join(base_mask_path, folder) + "/"
        imges = []
        masks = []
        for z in range(16):
            img = cv2.imread(src_path + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
            imges.append(img)
            masks.append(mask)

        test_imges = np.array(imges)
        test_imges = np.reshape(test_imges, (16, 96, 96))

        test_masks = np.array(masks)
        test_masks = np.reshape(test_masks, (16, 96, 96))

        Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",), inference=True,
                              model_path="log/segmeation/model/Vnet3d.pd-7000")
        # Vnet3d.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', dice_coefficient])

        predict = Vnet3d.prediction(test_imges)

        test_images = np.multiply(test_imges, 1.0 / 255.0)
        test_masks = np.multiply(test_masks, 1.0 / 255.0)
        predict_normalized = np.multiply(predict, 1.0 / 255.0)

        save_images(test_images, [4, 4], f"test_src_{folder}.bmp")
        save_images(test_masks, [4, 4], f"test_mask_{folder}.bmp")
        save_images(predict, [4, 4], f"test_predict_{folder}.bmp")

        # Calculate Dice coefficient
        dice_score = dice_coefficient(test_masks, predict_normalized) * 100

        # Calculate IoU
        iou_score = compute_iou(test_masks, predict_normalized) * 100

        # Calculate Sensitivity
        sensitivity_score = compute_sensitivity(test_masks, predict_normalized) * 100

        print(f"Folder: {folder}")
        print(f"Dice Coefficient (%): {dice_score}")
        print(f"IoU Score (%): {iou_score}")
        print(f"Sensitivity (%): {sensitivity_score}")

        results.append({
            "Folder": folder,
            "Dice Coefficient (%)": dice_score,
            "IoU Score (%)": iou_score,
            "Sensitivity (%)": sensitivity_score
        })

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel("segmentation_results_4.xlsx", index=False)

predict()
