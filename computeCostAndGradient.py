from cmath import log
from dataclasses import replace
import math
import numpy as np
from predictValue import predictValue


def computeCostAndGradient(D, Y, Hypothesis):
    m = D.shape[0]    #D.shape[0] = |rows| = number of examples
    D = np.array(D)
    Gradient = computeGradient(D, Y, Hypothesis, m)
    J = computeCost(D, Y, Hypothesis, m)
    return [Gradient, J]

def computeGradient(Data, Y, Hypothesis, examplesNum):
    gradient = list()
    errors = list()
    featuresNum = Data.shape[1]
    for i in range(examplesNum): 
        valuePredicted = predictValue(Data[i], Hypothesis)  #D[i,:] = whole row at index i
        valuePredicted = changeIfZero(valuePredicted)
        errors.append(valuePredicted-Y[i])   
    for j in range(featuresNum):
        gradient.append(np.sum(errors * Data[:,j]))
        gradient[j] = gradient[j]/examplesNum
    return np.array(gradient)

def computeCost(Data, Y, Hypothesis, examplesNum):
    costs = list()
    for i in range(examplesNum): 
        valuePredicted = predictValue(Data[i, :], Hypothesis)    
        valuePredicted = changeIfZero(valuePredicted)
        negativeValuePred = 1-valuePredicted
        costs.append((-Y[i] * math.log(valuePredicted))-((1 - Y[i]) * math.log(negativeValuePred)))
    finalCost = np.sum(costs)/examplesNum
    return finalCost

def changeIfZero(valuePredicted):
    ifZero = 0.0001 #not pass 0 to log
    if valuePredicted == 0:
        return ifZero
    elif valuePredicted == 1:
        return (1 - ifZero)
    return valuePredicted