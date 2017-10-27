import sys
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

def normalize(X, testX):
    allX = np.concatenate((X, testX)).astype(float)
    mu = sum(allX) / len(allX)
    sigma = np.std(allX, axis=0)
    
    for i in range(len(allX)):
        allX[i]=(allX[i]-mu)/sigma
    return allX[0:len(X)], allX[len(X):]

if __name__=='__main__':
    np.random.seed(7)

    col_name=[]
    X=[]
    with open(sys.argv[1], 'r') as f:
        col_name = f.readline()
        col_name = col_name.strip('\n').split(',')
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            X.append(line)
    X=np.array(X)
    XMAX=X.max(axis=0)

    Y=[]
    with open(sys.argv[2], 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            Y.append(line)
    Y=np.array(Y).astype('int32')
    Y=np.ravel(Y)

    testX=[]
    with open(sys.argv[3],'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX=np.array(testX)
    
    # Normalization
    X, testX = normalize(X, testX)
    
    model = XGBClassifier()
    model.fit(X, Y)
    y_ = model.predict(testX)
    
    print('id,label')
    for i,v in enumerate(y_):
        print('%d,%d'%(i+1,v))
