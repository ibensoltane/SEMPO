import numpy as np
import pandas as pd

c = 3e8
hb = 6.582119570e-1


def ReadDataFile(dataFilePath, columnNameIndex=None, FreqColName=None, DataColNameR=None, DataColNameI=None,
                 sep=",", decimal=".", useLambda=False, FreqConversion=1e-15, w1=-1, w2=500, nPts=75):
    a = pd.read_csv(dataFilePath, header=columnNameIndex, delimiter=sep, decimal=decimal)
    X = a.iloc[:, 0] if FreqColName is None else a[FreqColName]
    W = np.flip(2 * np.pi * c / X.to_numpy()).copy().astype(float) if useLambda else X.to_numpy().astype(float)
    W *= FreqConversion

    HwR = a.iloc[:, 1] if DataColNameR is None else a[DataColNameR]
    HwR = np.flip(HwR.to_numpy()).copy() if useLambda else HwR.to_numpy()
    HwI = a.iloc[:, 2] if DataColNameI is None else a[DataColNameI]
    HwI = np.flip(HwI.to_numpy()).copy() if useLambda else HwI.to_numpy()
    Hw = HwR + 1j * HwI

    Hw = Hw[np.argsort(W)]
    W = W[np.argsort(W)]

    Hw = Hw[(w1 < W) & (W < w2)]
    W = W[(w1 < W) & (W < w2)]

    if W.shape[0] > nPts:
        q = np.unique(np.round(np.linspace(0, W.shape[0] - 1, num=nPts)).astype(int))
        Hw = Hw[q]
        W = W[q]

    return W, Hw