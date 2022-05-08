#Fardous abu zaid - 211965108 , Itai simhony - 312179906
from plotDecisionBoundary import plotDecisionBoundary
from computeCostAndGradient import computeCostAndGradient
from updateHypothsis import updateHypothsis

def gradientDescent(Data, Y, Hypothesis, alpha, max_iter, threshold):
    plotDecisionBoundary(Hypothesis, Data, Y)
    Costs = list()
    i=0
    Costs.append(float('inf'))
    for i in range(max_iter):
        [gradient, cost] = computeCostAndGradient(Data, Y, Hypothesis)
        Costs.append(cost)
        if i > 0:
            improvement = abs(Costs[i-1]-Costs[i])
            if improvement < threshold:
                print(f'Gradient descent terminating after {i} iterations. Improvement was : {improvement} - below threshold ( {threshold} )')
                break
        Hypothesis = updateHypothsis(Hypothesis, alpha, gradient)
    if i >= max_iter:
        print("Gradient descent terminating after %s iterations (max_iter)" % i)
    plotDecisionBoundary(Hypothesis, Data, Y)
    return [Hypothesis, Costs]