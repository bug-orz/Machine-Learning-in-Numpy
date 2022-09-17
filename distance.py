import numpy as np
def Euclidean_Distance(x1,x2):
    return -np.linalg.norm(x1-x2,ord=2,axis=1)

def Manhattan_Distance(x1,x2):
    return -np.linalg.norm(x1-x2,ord=1,axis=1)

def Cosine_Distance(x1,x2):
    return (np.dot(x1, np.transpose(x2))/(np.linalg.norm(x1)*np.linalg.norm(x2)))[0]