import numpy as np
from sigmoid import sigmoid

def predictValue(Example, Hypothesis):
    Example = np.array(Example)
    Hypothesis = np.array(Hypothesis)
    prediction = np.sum(Example * Hypothesis)
    return sigmoid(prediction)
