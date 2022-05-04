import numpy as np
from sigmoid import sigmoid

def predictedValue(Example, Hypothesis):
    prediction = np.sum(Example * Hypothesis)
    return sigmoid(prediction)