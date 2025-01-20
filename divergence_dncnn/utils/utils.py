
import torch
import torch.nn as nn
import numpy as np

# ---- load the model based on the type and sigma (noise level) ---- 
def load_model(model_type, sigma):
    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    model_type == "DnCNN"
    from model.models import DnCNN
    net = DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net).cuda()


    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# ---- calculating PSNR (dB) of x ----- 
def psnr(x,im_orig):
    xout = (x - np.min(x)) / (np.max(x) - np.min(x))
    norm1 = np.sum((np.absolute(im_orig)) ** 2)
    norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    psnr = 10 * np.log10( norm1 / norm2 )
    return psnr
