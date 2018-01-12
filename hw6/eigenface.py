import numpy as np
from skimage.io import *

import sys, os

dir_path = sys.argv[1]
input_path = sys.argv[2]
output_path = 'reconstruction.jpg'

fname_list = list(os.listdir(dir_path))

X = np.array([imread(os.path.join(dir_path, fname)).flatten() \
              for fname in fname_list])

mu = np.mean(X, 0)
ma_data = X - mu

U, s, V = np.linalg.svd(ma_data.T, full_matrices=False)
e_faces = U
weights = np.dot(ma_data, e_faces)

def rescale(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    
    return M

def recon_with_k(fname, k=4):
    img_idx = fname_list.index(fname)
    return mu + np.dot(weights[img_idx, :k], e_faces.T[:k])

def to_img(I):
    return rescale(I).reshape(600,600,3)

recon = to_img(recon_with_k(input_path, k=4))
imsave(output_path, recon)
