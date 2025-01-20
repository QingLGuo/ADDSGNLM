# Utility functions required for denoisers
import torch
import numpy as np
import torch.nn.functional as F



def integral_img_sq_diff(v,dx,dy):
    t = img2Dshift(v,dx,dy)
    diff = (v-t)**2
    sd = torch.cumsum(diff,dim=0)
    sd = torch.cumsum(sd,dim=1)
    return(sd,diff,t)

def triangle(dx,dy,Ns):
    r1 = np.abs(1 - np.abs(dx)/(Ns+1))
    r2 = np.abs(1 - np.abs(dy)/(Ns+1))
    return r1*r2

def img2Dshift(v,dx,dy):
    row,col = v.shape[-2],v.shape[-1]
    t = torch.zeros_like(v)
    typ = (1 if dx>0 else 0)*2 + (1 if dy>0 else 0)
    if(typ==0):
        t[-dx:,-dy:] = v[0:row+dx,0:col+dy]
    elif(typ==1):
        t[-dx:,0:col-dy] = v[0:row+dx,dy:]
    elif(typ==2):
        t[0:row-dx,-dy:] = v[dx:,0:col+dy]
    elif(typ==3):
        t[0:row-dx,0:col-dy] = v[dx:,dy:]
    return t

def laplacian_filter(input_image):
    laplacian_kernel = torch.tensor([[1,1,1],
                                     [1,-8,1],
                                     [1,1,1]],dtype=torch.float32)

    kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)

    output_image = F.conv2d(input_image,kernel.to(input_image.device),padding=1)

    output_abs = torch.abs(output_image)
    output_mean = torch.mean(output_abs)
    output = 1-output_mean/8

    # output = 1-torch.sqrt(output_mean/8)
    # output = 1 - (output_mean / 8)**2
    # output = 1 - torch.pow(output_mean / 8,0.25)
    # print('output',output)
    # print(output_mean/8)
    # output = torch.exp(1-output_mean/8)/(torch.exp(torch.tensor(1.0))-1)
    # output = torch.exp(-output_mean/8)


    # output = 1-output_abs/8
    # print(output.shape)
    return output