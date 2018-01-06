import numpy as np
from skimage.io import *

import sys, os

dir_path = sys.argv[1]
input_path = sys.argv[2]
output_path = 'reconstruction.jpg'

X = np.array([imread(os.path.join(dir_path, '%d.jpg'%i)).flatten() \
              for i in range(415)])

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

def recon_with_k(img_idx, k=4):
    return mu + np.dot(weights[img_idx, :k], e_faces.T[:k])

def to_img(I):
    return rescale(I).reshape(600,600,3)

img_idx = int(input_path.split('.')[0])
recon = to_img(recon_with_k(img_idx, k=4))
imsave(output_path, recon)
