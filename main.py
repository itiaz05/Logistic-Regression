import numpy as np
from loadData import loadData
from sigmoid import sigmoid
from computeCostAndGradient import computeCostAndGradient
from addOnesCol import addOnesCol

if __name__ == '__main__':
    [D, Y] = loadData("Logistic-Regression/files/ex2data1.txt")
    D = addOnesCol(D)
    helpMat = np.ones((3,3),dtype=np.float64)*97
    #helpMat = [[-71.0, -70.0, -24.0], [-5.0, -65.0, -75.0], [-91.0, -73.0, -30.0]]
    #helpMat = [15.0, 24.5, 26.0]
    print(computeCostAndGradient(D, Y, [-10, 0.8, 0.08]))

