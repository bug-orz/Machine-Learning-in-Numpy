import numpy as np
from tqdm import tqdm,trange
from dataset import get_MNIST
from distance import Euclidean_Distance,Manhattan_Distance,Cosine_Distance

batch=1000 # Must be divisible by 60000 and 10000

train_x,train_y,val_x,val_y=get_MNIST()
#dist=Cosine_Distance
#dist=Euclidean_Distance
dist=Manhattan_Distance

def KNN(x,dist=dist,k=3):
    neighbors=np.array([])
    for i in range(int(train_x.shape[0]/batch)):
        neighbors=np.append(neighbors,dist(np.tile(x,(batch,1)), train_x[i*batch:(i+1)*batch]),axis=0)
    pred_index=np.argpartition(neighbors, -k)[-k:]
    return np.argmax(np.bincount(train_y[pred_index]))

def Accuracy(size=-1):
    right=0;wrong=0
    for i in trange(len(val_x[:size])):
        pred=KNN(val_x[i])
        if pred==val_y[i]:
            right+=1
        else:
            wrong+=1
    print("Accuracy: ",right/(right+wrong))

Accuracy(50)