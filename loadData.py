import string
import numpy as np
import matplotlib as pl

def loadData(filename):
    rows = 0
    try:
        with open(filename) as f:   #reading file
            D = f.read()
        D = D.translate(str.maketrans('', '', string.punctuation))  #clean input
        cleanedInputPath = "Logistic-Regression/files/cleanedInput.txt"    #path to save 
        with open(cleanedInputPath, 'w') as f:  #saving cleaned file for opening with numpy
            f.write(D)
        B = np.genfromtxt(cleanedInputPath)
        lastCol = np.size(B,1)-1 #index of last column
        Y = B[:,lastCol]    #Y is the predicated values
        B = B[:,0:lastCol]
        rows = B.shape[0]
    except Exception as e:
        print("Problem accrued reading the file: /n %s" %e)
    finally:
        print("read %s rows" % rows)
    return [B, Y]

