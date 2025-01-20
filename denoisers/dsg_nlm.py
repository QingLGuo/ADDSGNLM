# Kernel denoisers with guide image
import torch
import numpy as np
from .utils import *
import torch.nn.functional as F

def DSGNLM(noisy_img, guide_img, patch_rad, window_rad, sigma):
    if len(noisy_img.shape) != 4:  # Check if input is 4-dimensional (batch, channels, height, width)
        raise ValueError('Input must be a 4D array (batch, channels, height, width)')
    batch_size, num_channels, height, width = noisy_img.shape
    # u = torch.zeros((batch_size, num_channels, height, width)).to('cuda:0')
    u = torch.zeros((batch_size,num_channels,height,width)).to('cpu')


    for j in range(batch_size):

        guide_img_new = guide_img[j].unsqueeze(0)
        noisy_img_new = noisy_img[j].unsqueeze(0)

        padded_guide = F.pad(guide_img_new, (patch_rad, patch_rad, patch_rad, patch_rad), mode='reflect')
        padded_v = F.pad(noisy_img_new, (window_rad, window_rad, window_rad, window_rad),mode='reflect')

        padded_guide = padded_guide.squeeze(0)
        padded_v = padded_v.squeeze(0)


        # 0th loop
        # W0 = torch.zeros((height, width)).to('cuda:0')
        W0 = torch.zeros((height, width)).to('cpu')
        for dx in np.arange(-window_rad, window_rad + 1):
            for dy in np.arange(-window_rad, window_rad + 1):
                sd, diff, t = integral_img_sq_diff(padded_guide.squeeze(0), dx, dy)
                # hat = triangle(dx, dy, window_rad)
                temp1 = img2Dshift(sd, patch_rad, patch_rad)
                temp2 = img2Dshift(sd, -patch_rad - 1, -patch_rad - 1)
                temp3 = img2Dshift(sd, -patch_rad - 1, patch_rad)
                temp4 = img2Dshift(sd, patch_rad, -patch_rad - 1)
                res = temp1 + temp2 - temp3 - temp4
                sqdist1 = res[patch_rad:patch_rad + height, patch_rad:patch_rad + width]
                # w = hat * torch.exp(-sqdist1 / (sigma ** 2))

                w = torch.exp(-sqdist1 / (sigma ** 2))
                W0 = W0 + w

        # 1st loop
        # W1 = torch.zeros((height, width)).to('cuda:0')
        W1 = torch.zeros((height, width)).to('cpu')
        for dx in np.arange(-window_rad, window_rad + 1):
            for dy in np.arange(-window_rad, window_rad + 1):
                sd, diff, t = integral_img_sq_diff(padded_guide.squeeze(0), dx, dy)
                # hat = triangle(dx, dy, window_rad)
                temp1 = img2Dshift(sd, patch_rad, patch_rad)
                temp2 = img2Dshift(sd, -patch_rad - 1, -patch_rad - 1)
                temp3 = img2Dshift(sd, -patch_rad - 1, patch_rad)
                temp4 = img2Dshift(sd, patch_rad, -patch_rad - 1)
                res = temp1 + temp2 - temp3 - temp4
                sqdist1 = res[patch_rad:patch_rad + height, patch_rad:patch_rad + width]
                # w = hat * torch.exp(-sqdist1 / (sigma ** 2))
                w = torch.exp(-sqdist1 / (sigma ** 2))
                # W0_pad = np.pad(W0, window_rad, mode='symmetric')
                W0 = W0.unsqueeze(0).unsqueeze(0)
                W0_pad = F.pad(W0, (window_rad, window_rad, window_rad, window_rad), mode='reflect')


                W0 = W0.squeeze()
                W0_pad = W0_pad.squeeze()
                W0_shift = img2Dshift(W0_pad, dx, dy)
                W0_temp = W0_shift[window_rad:window_rad + height, window_rad:window_rad + width]
                w1 = w / (torch.sqrt(W0) * torch.sqrt(W0_temp))
                W1 = W1 + w1

        # 2nd loop
        alpha = 1 / torch.max(W1)
        # W2 = torch.zeros((height, width)).to('cuda:0')
        W2 = torch.zeros((height, width)).to('cpu')
        for dx in np.arange(-window_rad, window_rad + 1):
            for dy in np.arange(-window_rad, window_rad + 1):
                if ((dx != 0) or (dy != 0)):
                    sd, diff, t = integral_img_sq_diff(padded_guide.squeeze(0), dx, dy)
                    # hat = triangle(dx, dy, window_rad)
                    temp1 = img2Dshift(sd, patch_rad, patch_rad)
                    temp2 = img2Dshift(sd, -patch_rad - 1, -patch_rad - 1)
                    temp3 = img2Dshift(sd, -patch_rad - 1, patch_rad)
                    temp4 = img2Dshift(sd, patch_rad, -patch_rad - 1)
                    res = temp1 + temp2 - temp3 - temp4
                    sqdist1 = res[patch_rad:patch_rad + height, patch_rad:patch_rad + width]
                    # w = hat * torch.exp(-sqdist1 / (sigma ** 2))
                    w = torch.exp(-sqdist1 / (sigma ** 2))

                    W0 = W0.unsqueeze(0).unsqueeze(0)
                    W0_pad = F.pad(W0, (window_rad, window_rad, window_rad, window_rad), mode='reflect')
                    W0 = W0.squeeze()
                    W0_pad = W0_pad.squeeze()
                    W0_shift = img2Dshift(W0_pad, dx, dy)
                    W0_temp = W0_shift[window_rad:window_rad + height, window_rad:window_rad + width]
                    w2 = (alpha * w) / (torch.sqrt(W0) * torch.sqrt(W0_temp))
                    padded_v = padded_v.squeeze(0)
                    v = padded_v[window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                    u[j] = u[j] + w2 * v
                    W2 = W2 + w2

        u[j] = u[j] + (1 - W2) * noisy_img[j]
    return u


