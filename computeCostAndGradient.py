from math import log
import numpy as np
from computeRegularizedCostAndGradient import computeRegularizedCostAndGradient
from predictValue import predictValue

def computeCostAndGradient(D, Y, Hypothesis):
    lambdaValue = 1000
    D = np.array(D)
    m = D.shape[0]    #D.shape[0] = |rows| = number of examples
    J = computeCost(D, Y, Hypothesis, m)
    Gradient = computeGradient(D, Y, Hypothesis, m)
    [costRegularFactor, gradientRegularFactor] = computeRegularizedCostAndGradient(D, Y, Hypothesis, lambdaValue)
    regularizedGradient = np.add(Gradient, gradientRegularFactor)
    regularizedCost = J + costRegularFactor
    return [regularizedGradient, regularizedCost]

def computeGradient(Data, Y, Hypothesis, examplesNum):
    gradient = list()
    errors = list()
    featuresNum = Data.shape[1]
    for i in range(examplesNum): 
        valuePredicted = predictValue(Data[i], Hypothesis)  #D[i,:] = whole row at index i
        valuePredicted = changeIfZero(valuePredicted)
        errors.append(valuePredicted-Y[i])   
    for j in range(featuresNum):
        gradientCell = np.sum(errors * Data[:,j])
        gradientCell /= examplesNum
        gradient.append(gradientCell)
    return np.array(gradient)

def computeCost(Data, Y, Hypothesis, examplesNum):
    costs = list()
    for i in range(examplesNum): 
        valuePredicted = predictValue(Data[i, :], Hypothesis)    
        valuePredicted = changeIfZero(valuePredicted)
        negativeValuePred = 1-valuePredicted
        costs.append((-Y[i] * log(valuePredicted))-((1 - Y[i]) * log(negativeValuePred)))
    finalCost = np.sum(costs)/examplesNum
    return finalCost

def changeIfZero(valuePredicted):
    ifZero = 0.0001 #not pass 0 to log
    if valuePredicted == 0:
        return ifZero
    elif valuePredicted == 1:
        return (1 - ifZero)
    return valuePredicted


