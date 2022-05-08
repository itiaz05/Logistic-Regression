import numpy as np


def computeRegularizedCostAndGradient(D, Y, Hypothesis, lambdaValue):
    examplesNum = D.shape[0]
    costFactor = costRegularizationFactor(Hypothesis, examplesNum, lambdaValue)
    gradientFactor = gradientRegularizationFactor(Hypothesis, examplesNum, lambdaValue)    
    return [costFactor, gradientFactor]

def costRegularizationFactor(Hypothesis, examplesNum, lambdaValue):
    hypothesisSum = 0
    Hypothesis = np.array(Hypothesis)
    for i in range(1, Hypothesis.size):
        hypothesisSum += Hypothesis[i] * Hypothesis[i]
    return  hypothesisSum * lambdaValue / (2 * examplesNum)

def gradientRegularizationFactor(Hypothesis, examplesNum, lambdaValue):
    regularizedGradient = list()
    gradientSize = np.array(Hypothesis).size
    for j in range(gradientSize):
        if j==0:
            regularizedGradient.append(0)
        else:
            regularizedGradient.append((lambdaValue * Hypothesis[j]) / examplesNum)
    return np.array(regularizedGradient)
