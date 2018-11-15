import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import itertools, imageio, random



def to_img(X):
    X = 0.5 * (X + 1)
    X = X.clamp(0, 1)
    return X

def show_im(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow(np.transpose (X.numpy(), (1,2,0)))
        plt.show()
    elif X.dim() == 2:
        plt.imshow(X.numpy(), cmap='gray')
        plt.show()
    else:
        print('WRONG TENSOR SIZE of show_im')
        
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
    
def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0    # record last index
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:    # choose the some samples (delete train or test, according to subfolder)
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)