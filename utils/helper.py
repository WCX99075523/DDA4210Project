'''
@misc{andersson2020generative,
      title={Generative Adversarial Networks for photo to Hayao Miyazaki style cartoons}, 
      author={Filip Andersson and Simon Arvidsson},
      year={2020},
      eprint={2005.07702},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
Edge promoting code with parallel computing. This piece of code is written by Filip Andersson & Simon Arvidsson
We directly use there implementation.

original paper: https://doi.org/10.48550/arXiv.2005.07702
project dir: https://github.com/FilipAndersson245/cartoon-gan
'''

import os
import cv2
import tqdm
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing.dummy as mp

from numba import njit, jit
from PIL import Image, ImageOps
from torchvision import transforms

n_threads = 8
p = mp.Pool(n_threads)

@jit
def edge_promoting(root, save):
    img_size = (384, 384)
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    
    pbar = tqdm.tqdm(total=len(file_list))
    
    job_args = [(os.path.join(root, f), gauss, img_size, kernel, kernel_size, save, n) for n, f in enumerate(file_list)]

    for _ in p.imap_unordered(edge_job, job_args):       
        pbar.update(1)

@njit()
def fast_loop(gauss_img, pad_img, kernel_size, gauss, dilation):
    idx = np.where(dilation != 0)
    loops = int(np.sum(dilation != 0))
    for i in range(loops):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
    return gauss_img

@jit
def edge_job(args):
    output_size = 256, 256
    path, gauss, img_size, kernel, kernel_size, save, n = args
    rgb_img = cv2.imread(path)
    gray_img = cv2.imread(path, 0)
    if rgb_img is None:
        print(path, "Error!")
        return
    rgb_img = np.array(ImageOps.fit(Image.fromarray(rgb_img), img_size, Image.ANTIALIAS))
    pad_img = np.pad(rgb_img, ((3, 3), (3, 3), (0, 0)), mode='reflect')
    gray_img = np.array(ImageOps.fit(Image.fromarray(gray_img), img_size, Image.ANTIALIAS))
    edges = cv2.Canny(gray_img, 150, 500) 
    dilation = cv2.dilate(edges, kernel)

    _gauss_img = np.copy(rgb_img)
    gauss_img = fast_loop(_gauss_img, pad_img, kernel_size, gauss, dilation)

    rgb_img = cv2.resize(rgb_img, output_size, Image.ANTIALIAS)
    gauss_img = cv2.resize(gauss_img, output_size)
    comb_img = np.concatenate((rgb_img, gauss_img), axis=1)
    cv2.imwrite(os.path.join(save, str(n) + '.jpg'), comb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def save_model(model,dir,use_pickle=False):
    if use_pickle:
        with open(dir, "rb") as f:
            pickle.dump(model.parameters(),f)
            f.close()
    else:
        torch.save(model,dir)

def load_model(dir):
    return torch.load(dir)

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

import cv2

def change_size(read_file):
    image = cv2.imread(read_file, 1)  
    img = image
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY) 
    binary_image = b[1] 
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  
    right = max(edges_x)  
    width = right - left  
    bottom = min(edges_y) 
    top = max(edges_y)  
    height = top - bottom  
    pre1_picture = image[left:left + width, bottom:bottom + height] 
    return pre1_picture 
