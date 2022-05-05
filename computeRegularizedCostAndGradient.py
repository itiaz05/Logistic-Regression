import numpy as np


def computeRegularizedCostAndGradient(D, Y, Hypothesis, gradient, cost, lambdaValue):
    examplesNum = calculateSize(D)
    regularizedCost = computeRegularizedCost(Hypothesis, examplesNum, lambdaValue)
    regularizedGradient = computeRegularizedGradient(gradient, Hypothesis, examplesNum, lambdaValue)
    regularizedCost += cost
    return [regularizedCost, regularizedGradient]

def computeRegularizedCost(Hypothesis, examplesNum, lambdaValue):
    Hypothesis = np.array(Hypothesis)
    Hypothesis = Hypothesis[1:,]**2 
    hypothesisSum = np.sum(Hypothesis)
    return  (hypothesisSum * lambdaValue) / (2 * examplesNum)

def computeRegularizedGradient(gradient, Hypothesis, examplesNum, lambdaValue):
    gradientSize = calculateSize(gradient)
    for j in range(gradientSize):
        if not j==0:
            gradient[j] += (lambdaValue * Hypothesis[j]) / examplesNum
    return  gradient

def calculateSize(arr):
    try:
        return arr.shape[0]
    except AttributeError:
        arr = np.array(arr)
        return arr.shape[0]
    except Exception as e:
        return False

