import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import cv2
from Architecture.vnet_architecture import Vnet3dModule
import numpy as np
from Architecture.layer import save_images


def predict():
    src_path = "segmentation/Image/1_5/"
    imges = []
    for z in range(16):
        img = cv2.imread(src_path + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
        imges.append(img)

    test_imges = np.array(imges)
    test_imges = np.reshape(test_imges, (16, 96, 96))

    Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log/segmentation/model/Vnet3d.pd-2000")
    predict = Vnet3d.prediction(test_imges)
    test_images = np.multiply(test_imges, 1.0 / 255.0)
    save_images(test_images, [4, 4], "test_src_2.bmp")
    save_images(predict, [4, 4], "test_predict_2.bmp")


predict()
