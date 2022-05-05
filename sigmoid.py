import math
import numpy as np


def sigmoid(z):
    if np.ndim(z) == 0: #dimension = 0 => is a scalar (not array/list)
        return g(z)
    z = np.array(z)
    sigmoidMat = np.array(z)
    isMatrix = False
    try:
        columns = z.shape[1]  # dimension > 1 => z is matrix
        isMatrix = True
    except IndexError: #z is 1 dimension array/list
        isMatrix = False
    except TypeError:
        print("there is a problem with data type, please try again")
    if isMatrix:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                sigmoidMat[i, j] = g(z[i, j])
    else:   #z is an array
        for i in range(z.shape[0]):
            sigmoidMat[i] = g(z[i])
    return sigmoidMat

def g(z):
    try:
        answer = 1 / (1+ math.exp(-z))
    except Exception as e:
        answer = 0  #math error / number too big/small to represent
    finally:
        return answer

