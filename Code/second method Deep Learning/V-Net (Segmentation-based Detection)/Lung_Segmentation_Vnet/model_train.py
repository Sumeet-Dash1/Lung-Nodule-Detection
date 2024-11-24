import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Architecture.vnet_architecture import Vnet3dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('train_X_mask.csv')
    csvimagedata = pd.read_csv('train_X.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",))
    Vnet3d.train(imagedata, maskdata, "Vnet3d.pd", "log\\segmentation\\", 0.001, 0.5, 10, 6)


train()
