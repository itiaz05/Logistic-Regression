from cmath import log
from dataclasses import replace
import math
import numpy as np
from predictedValue import predictedValue


def computeCostAndGradient(D, Y, Hypothesis):
    replace0 = 0.0001   #not pass 0 to log
    D = np.array(D)
    examplesNum = D.shape[0]    #m
    J = list()
    errors = list()
    gradient = list()
    for i in range(D.shape[0]): #D.shape[0] = number of rows 
        print("----------- iteration number %i -------------" % i)
        valuePredicted = predictedValue(D[i, :], Hypothesis)    #D[i,:] = whole row at index i
        if valuePredicted == 0:
            valuePredicted = replace0
        elif valuePredicted == 1:
            valuePredicted -= replace0 
        negativeValuePred = 1-valuePredicted
        errors.append(valuePredicted - Y[i])
        negativeValuePred = 1-valuePredicted  
        J.append((-Y[i] * math.log(valuePredicted))-((1 - Y[i]) * math.log(negativeValuePred)))
    #for j in range(D.shape[1]):
    #    gradient.append()
    finaleJ = np.sum(J)/examplesNum
    return finaleJ
