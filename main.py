#Fardous abu zaid - 211965108 , Itai simhony - 312179906
import numpy as np
from loadData import loadData
from computeCostAndGradient import computeCostAndGradient
from addOnesCol import addOnesCol
from gradientDescent import gradientDescent

if __name__ == '__main__':
    alpha = 0.001;
    max_iter = 1000;
    threshold = 0.0001;
    Hypothesis = [-10, 0.8, 0.08]
    #Hypothesis = [-8, 2, -0.5]
    filename = "Logistic-Regression/files/ex2data1.txt"
    [D, Y] = loadData(filename)
    D = addOnesCol(D)
    [finalHypothesis, Costs] = gradientDescent(D, Y, Hypothesis, alpha, max_iter, threshold)
    print("final cost = " ,Costs[len(Costs)-1])
    print("final hypothesis = " ,finalHypothesis)
    
