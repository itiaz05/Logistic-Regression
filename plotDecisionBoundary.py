from matplotlib import pyplot as plt
import numpy as np

def matchIndex(arrayToDisplay, index):
    try:
        indexMatched = arrayToDisplay.index(index)
        return indexMatched
    except:
        return -1

def plotDecisionBoundary(theta, X, y):
    negatives = np.empty((0, 2))
    positives = np.empty((0, 2))
    minX = np.min(X[:, 2])
    maxX = np.max(X[:, 2])
    plotX = np.array([minX - 2, maxX + 2])
    i = matchIndex(theta, 0)
    if i != -1:
        theta[i] = 0.001
    plotY = (-1 / theta[2]) * (theta[1] * plotX + theta[0])
    for x in range(X.shape[0]):
        firstValue = X.item((x, 1))
        secondValue = X.item((x, 2))
        line = np.array([[firstValue, secondValue]])
        if y[x] == 0:
            negatives = np.append(negatives, line, axis=0)
        else:
            positives = np.append(positives, line, axis=0)
    plt.scatter(positives[:, 0], positives[:, 1],   s=25,
                c='b', marker="+", label='Positive')
    plt.scatter(negatives[:, 0], negatives[:, 1],  s=25,
                c='r', marker=".", label='Negative')
    plt.plot(plotX, plotY,  c="m", label="Decision Boundary")
    plt.legend(loc='upper right')
    plt.show()