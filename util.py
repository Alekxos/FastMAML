import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

def accuracy(predictions, target):
    assert(predictions.shape == target.shape)
    total_entries = np.prod(list(predictions.shape))
    return torch.sum(torch.eq(predictions, target)) / total_entries

def display_image(image_array):
    plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
    plt.show()

def ConvBlock(id, input_channels, output_channels=64, conv_kernel_size=3, batch_norm_features=64, pool_kernel_size=2, pool_stride=2, momentum=1, affine=True):
    return [
        (f'conv{id}', nn.Conv2d(input_channels, output_channels, conv_kernel_size)),
        (f'bn{id}', nn.BatchNorm2d(batch_norm_features, momentum=momentum, affine=affine)),
        (f'relu{id}', nn.ReLU(inplace=True)),
        (f'pool{id}', nn.MaxPool2d(pool_kernel_size, pool_stride))
    ]