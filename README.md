# SEMPO

## Requirements
To use SEMPO, you will need to install some packages:
```
torch (from the official website)
numpy
tqdm
pandas
seaborn
matplotlib
visdom
```

## Installation
Simply run the following command in a console to install with pip
```
pip install SEMPO
```

## Documentation
A list of functionalities can be found in the following document: https://github.com/ibensoltane/SEMPO/blob/main/doc/Documentation.pdf


## TUTORIAL

Setup the example


```python
import visdom
import numpy as np

from SEMPO import Autodiff as SPA
from SEMPO import Cauchy as SPC
from SEMPO import DataManager as SPD
from SEMPO import Model as SPM
from SEMPO import ParameterManager as SPP
from SEMPO import Results as SPR

c = 3e8
hb = 6.582119570e-1

```

Load the data


```python
W, Hw = SPD.ReadDataFile("data/reseauOr.csv", DataColNameR="re", DataColNameI="im", columnNameIndex=0,
                         sep=";", FreqConversion=1e-6, useLambda=True, w1=0, w2=5, nPts=75)
```

Plot the Bode diagram to check the data


```python
fig = SPR.BodeDiagram(Hw, W, figID=1, fs=[5,4])
fig.subplots_adjust(hspace=.3, bottom=.17, top=.96, left=.15, right=.96)
fig.savefig("figures/tuto/1.png", dpi=400)
```


    
![img1](https://github.com/ibensoltane/SEMPO/blob/main/imgs/output_6_0.png?raw=True)
    


Use the ADC method to retrieve poles and zeros


```python
g0, p, z = SPC.CauchyMethodOptZP(Hw, W, nmbMaxPoles=10, dWorigin=1e-3, phys=True, stability=False,
                                 qStability=0., diffZPMax=4, useLogPrec=False, plotSingularValues=False)
```

Use the resulting g0, p & z to calculate the approximation


```python
Yw = SPM.G_SZF(W, g0, p, z)
```

Plot the amplitude to diagram to check the data


```python
fig, ax = SPR.AmplitudeCurve(Yw, W, figID=2, fs=[5,4])
fig.subplots_adjust(hspace=.3, bottom=.17, top=.96, left=.15, right=.96)
ax.scatter(x=W, y=np.abs(Hw), color="red", s=12, marker="x", zorder=200)
ax.legend(["Cauchy", "Sim. Data"])
fig.savefig("figures/tuto/2.png", dpi=400)
```


    
![img2](https://github.com/ibensoltane/SEMPO/blob/main/imgs/output_12_0.png?raw=True)
    


Adjust the results via autodiff with the SEMPO object parameterList


```python
parameterList = SPP.GenerateSZFParameterList(True, G0=g0, poles=p, zeros=z,
                                             nPoles=0, nPolesIm=0, nPolesEff=0, stableEffectivePoles=True, poleOrigin=False,
                                             nZeros=0, nZerosEff=0, nZerosIm=0, reversibleZeros=False,
                                             w=None, minDist=1e-4)
```

Convert to SEM to get poles and residues


```python
parameterList = SPP.ConvertToSEM(parameterList)
```

Optimize the SEM parameters ==> impose the Hermitian symmetry and try to reduce the error


```python
loss_viz = visdom.Visdom()
nIter = 50000
scIter = 400
scq = 0.91
lr = 5e-3
parameterList = SPA.FitModel(W=W, Ta=Hw, parameterList=parameterList, imaginaryPartPositive=False,
                             alpha=(1,0,0,0), tensorizeInput=True, nIter=nIter, lr=lr, scIter=scIter, scq=scq,
                             visualizeLoss=True, vizIteration=800, loss_viz=loss_viz, logScalePoles=True)
```

    Setting up a new session...
    Fitting SEM: 100%|██████████████████████████████████████████████████████████████| 50000/50000 [00:58<00:00, 857.51it/s]
    

Convert tensors to np array


```python
parameterList = SPP.Numpify(parameterList)
```

Use parameterList to calculate the approximation


```python
fctType, Hterms = SPA.H_terms(W, parameterList)
Yw2 = SPA.H(fctType, Hterms, tensorized=False)
```

Plot the amplitude to diagram to check the data


```python
fig, ax = SPR.AmplitudeCurve(Yw2, W, figID=3, fs=[5,4])
fig.subplots_adjust(hspace=.3, bottom=.17, top=.96, left=.15, right=.96)
ax.scatter(x=W, y=np.abs(Hw), color="red", s=12, marker="x", zorder=200)
ax.legend(["SEM", "Sim. Data"])
fig.savefig("figures/tuto/3.png", dpi=400)
```


    
![img3](https://github.com/ibensoltane/SEMPO/blob/main/imgs/output_24_0.png?raw=True)
    

