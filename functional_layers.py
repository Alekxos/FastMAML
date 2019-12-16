import numpy as np
import torch
from torch.nn import functional as F

def linear(input, weight, bias=None, cuda=True):
    if cuda:
        return F.linear(input, weight.cuda()) if bias is None else F.linear(input, weight.cuda(), bias.cuda())
    return F.linear(input, weight) if bias is None else F.linear(input, weight, bias)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, cuda=True):
    if cuda:
        return F.conv2d(input, weight.cuda(), bias.cuda(), stride, padding, dilation, groups)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

def relu(input):
    return F.threshold(input, 0, 0, inplace=True)

def max_pool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def batch_norm(input, weight=None, bias=None, training=True, epsilon=1.e-5, momentum=0.1, cuda=True):
    if cuda:
        running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
        running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
    else:
        running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).double()
        running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).double()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, epsilon)