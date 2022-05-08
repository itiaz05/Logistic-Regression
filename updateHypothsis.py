import numpy as np

def updateHypothsis(Hypothesis, alpha, Gradient):
    Hypothesis = np.array(Hypothesis)
    Gradient = np.array(Gradient)
    hypothesisSize = Hypothesis.size
    gradientSize = Gradient.size
    if hypothesisSize != gradientSize:
        print("vectors' length is not equal: gradient size=%s and hypothesis size=%s" % (gradientSize, hypothesisSize))
        return 0
    #for i in range(hypothesisSize):
    #    Hypothesis[i] -= alpha * Gradient[i]
    newGrad = Gradient * alpha
    newHypo = Hypothesis - newGrad
    return newHypo
    #TODO: 2 ways to figure out the best