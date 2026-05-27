#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a image with multiple channels and multiple kernels
using given padding and stride
"""


import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels
    using multiple kernels and given padding and stride
    """
    m, height, width, c = images.shape
    kh, kw, kc, nc = kernel.shape
    sh, sw = stride
    if padding is 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw, nc))
    for index in range(nc):
        kernel_index = kernel[:, :, :, index]
        i = 0
        for h in range(0, (height + (2 * ph) - kh + 1), sh):
            j = 0
            for w in range(0, (width + (2 * pw) - kw + 1), sw):
                output = np.sum(
                    images[:, h: h + kh, w: w + kw, :] * kernel_index,
                    axis=1).sum(axis=1).sum(axis=1)
                convoluted[:, i, j, index] = output
                j += 1
            i += 1
    return convoluted
