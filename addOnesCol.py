import numpy as np

def addOnesCol(D):
    D = np.array(D)
    newCol = np.ones((np.size(D,0),1))
    return np.hstack((newCol, D))