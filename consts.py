import os
import torch
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import sys
import json
DATASET_ROOT_DIR_PATH = "D:\magshimim\project\dataset"

"""##Consts"""

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
RGB_CHANNELS = 3
HEATMAP_CHANNELS = 21
STRIDE = 2
DECONV_STRIDE = 2
DECONV_KERNEL_SIZE = 2
MAX_POOL_FILTER = (2,2)
DOWNSCALE_PADDING = 3

HOURGLASS_OUTPUT_CHANNELS = 128
#HOURGLASS_OUTPUT_CHANNELS = 64
UPSCALE = 2
BATCH_NORM_MOMENTUM = 1e-5

# after every applied Conv filter- the image shrinks,
# if the images dimensions were (C, H, W), and padding=p, stride=s, kernel_size=f
# then the output would be - Hout = lower((H+2*p-f)/s + 1)
calc_out_size = (lambda h,p,s,f : int((h+2*p-f)/s + 1))

DATASET_PATH = os.path.join(DATASET_ROOT_DIR_PATH, "Dataset/Dataset")
LABEL_PATH = os.path.join(DATASET_ROOT_DIR_PATH, "Full_Data.csv")
DEFAULT_FILENAME = "model.bin"

# Loss
HAND_DETECTION_INDEX = 0  # besides the keypoint coordinates, the Model returns a bool representing whether a hand was detected in the
# image, it's index in the labels is 0
INVALID_DETECTION_LOSS_VALUE = 1  # if the model detects a hand incorrectly, the loss value should is 1, ignoring the actual difference


# HMs to KP
HEATMAP_SIZE = 64
BLUR_DESC = 3
BLUR_GRID_SIZE = 25
BLUR_GRID = (BLUR_GRID_SIZE, BLUR_GRID_SIZE)
MAX_VALUE = 1
MIN_VALUE = 0

N_KEYPOINTS = 21


# Conv Block

KERNEL_SIZE = 7
PADDING = 3  # if we want the output's height to be half of the input's height(while the kernel_size is 7),
#  we need the padding to be 3 - based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
SAME_PADDING = 67


# Sizes

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Cuda


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Dataset

def resize_transform(img):
    return cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

def scale_transform(img):
    return np.cast['float32'](img) / 255.0  # the pixels in the image are scaled from 0-255 to 0-1)
# a normalization transformation can also be added


# Model

HOURGLASS_LAYERS = 2

# Training
WEIGHT_DECAY = 0
BATCH_SIZE = 10

INIT_LEARNING_RATE = 0.01



IMG = "1.jpg"

get_id_from_name = (lambda name: int(name.split(".")[0]))
