import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import sys, os
import pickle

img = np.load(sys.argv[1])
img = img / 255

TRAIN = False

if TRAIN:
    min_ = 140000
    for i in range(50):
        pca = PCA(n_components=280, whiten=True, svd_solver='randomized').fit_transform(img)

        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
        kmeans.fit(pca)

        sum_ = sum(kmeans.labels_)

        if abs(sum_-70000) <= min_:
            min_ = abs(sum_-70000) 
            label = kmeans.labels_
            print(i,sum_, label)
            if sum_ == 70000:
                break
                
    pickle.dump(label, open('label.pkl','wb'))

else:
    label = pickle.load(open('label.pkl','rb'))

df = pd.read_csv(sys.argv[2])
ii = df['ID']
id1= df['image1_index']
id2= df['image2_index']

with open(sys.argv[3], 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(ii)):
        if label[id1[i]] == label[id2[i]]:
            f.write('%d,1\n'%ii[i])
        else:
            f.write('%d,0\n'%ii[i])