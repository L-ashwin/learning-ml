import numpy as np
import pandas as pd

def testTrainSplit(xData, yData, Testfrac = 0.2, stratify = False):
    test, train = [], []

    if stratify:
        labels = np.unique(yData)
        for label in labels:
            ind, = np.where(yData == label)
            np.random.shuffle(ind)
            test.extend(ind[0:int(len(ind)*Testfrac)])
            train.extend(ind[int(len(ind)*Testfrac):])
    else:
        ind = np.arange(len(yData))
        np.random.shuffle(ind)
        test  = ind[0:int(len(ind) * Testfrac)]
        train = ind[int(len(ind) * Testfrac) :]

    xTest  = xData[test]
    xTrain = xData[train]
    yTest  = yData[test]
    yTrain = yData[train]
    return xTest, xTrain, yTest, yTrain
