import torch
import torch.optim as optim

import numpy as np
from tqdm import trange

from . import ParameterManager as SPP



c = 3e8
hb = 6.582119570e-1



def H_terms(w, parameterList):
    v = []
    w1 = w[..., None]

    fctType = parameterList[0]
    tensorized = parameterList[1]
    stableEffectivePoles = parameterList[-1]

    if fctType == "sem":
        Hnr, pnC_R, pnC_I, rnC_R, rnC_I, pnI_I, rnI_I, pnE_R, pnE_I, rnE_R, rnE_I, r0, poleOrigin = parameterList[2:-1]

        v.append(Hnr)

        if tensorized:
            pnC = torch.abs(pnC_R) - 1j*torch.abs(pnC_I)
            rnC = rnC_R + 1j*rnC_I

            pnI = -1j * torch.abs(pnI_I)
            rnI = 1j * rnI_I/2

            pnE = torch.abs(pnE_R) - 1j*torch.abs(pnE_I) if stableEffectivePoles else torch.abs(pnE_R) + 1j*pnE_I
            rnE = rnE_R + 1j * rnE_I

            if poleOrigin:
                v.append(-1j*torch.abs(r0)/w1)

            v.append(rnC/(w1-pnC) - rnC.conj()/(w1+pnC.conj()))
            v.append(rnI/(w1-pnI) - rnI.conj()/(w1+pnI.conj()))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))

        else:
            pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
            rnC = rnC_R + 1j * rnC_I

            pnI = -1j * np.abs(pnI_I)
            rnI = 1j * rnI_I / 2

            pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
            rnE = rnE_R + 1j * rnE_I

            if poleOrigin:
                v.append(-1j*np.abs(r0)/w1)

            v.append(rnC/(w1-pnC) - rnC.conj()/(w1+pnC.conj()))
            v.append(rnI/(w1-pnI) - rnI.conj()/(w1+pnI.conj()))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))


    elif fctType == "gdl":
        Hnr, G0, an, bn, s1n, Gn, wn, s2n, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[2:-1]

        v.append(1+Hnr)

        if tensorized:
            s1nn = s1n
            Gnn = 0.00005 + torch.abs(Gn)
            wnn = 0.00005 + torch.abs(wn)
            s2nn = s2n

            pnE = torch.abs(pnE_R) - 1j*torch.abs(pnE_I) if stableEffectivePoles else torch.abs(pnE_R) + 1j*pnE_I
            rnE = rnE_R + 1j * rnE_I

            v.append(1j*G0/w)
            v.append(-an**2/(w1**2 + 1j*torch.abs(bn)*w1))
            v.append(-(1j*s1nn*Gnn*w1 + s2nn*((Gnn/2)**2 + wnn**2))/((w1**2 - (Gnn/2)**2 - wnn**2) + 1j*Gnn*w1))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))

        else:
            s1nn = s1n
            Gnn = 0.00005 + np.abs(Gn)
            wnn = 0.00005 + np.abs(wn)
            s2nn = s2n

            pnE = np.abs(pnE_R) - 1j*np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j*pnE_I
            rnE = rnE_R + 1j * rnE_I

            v.append(1j*G0/w)
            v.append(-an**2/(w1**2 + 1j*np.abs(bn)*w1))
            v.append(-(1j*s1nn*Gnn*w1 + s2nn*((Gnn/2)**2 + wnn**2))/((w1**2 - (Gnn/2)**2 - wnn**2) + 1j*Gnn*w1))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))


    elif fctType == "mtse":
        Hnr, mrn, trn, wnR, wnI, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[2:-1]

        v.append(Hnr)

        if tensorized:
            rn = torch.abs(mrn) * torch.exp(1j * trn)
            wn = torch.abs(wnR) - 1j * torch.abs(wnI)

            Cn = -rn / (2j * torch.imag(wn))
            Tn1 = 1j * torch.log((w1 - wn) / (w1 - wn.conj())) + trn + torch.pi/2
            Tn2 = 1j * torch.log((w1 + wn.conj()) / (w1 + wn)) - trn - torch.pi/2

            pnE = torch.abs(pnE_R) - 1j*torch.abs(pnE_I) if stableEffectivePoles else torch.abs(pnE_R) + 1j*pnE_I
            rnE = rnE_R + 1j * rnE_I

            v.append(2 * torch.real(Cn) + torch.abs(Cn) * (torch.exp(1j * Tn1) + torch.exp(1j * Tn2)))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))

        else:
            rn = np.abs(mrn) * np.exp(1j * trn)
            wn = np.abs(wnR) + 1j * wnI

            Cn = -rn / (2j * np.imag(wn))
            theta = trn - np.pi / 2 + np.pi * (wnI > 0)
            Tn1 = 1j * np.log((w1 - wn) / (w1 - wn.conj())) + theta + np.pi
            Tn2 = 1j * np.log((w1 + wn.conj()) / (w1 + wn)) - theta + np.pi

            pnE = np.abs(pnE_R) - 1j*np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j*pnE_I
            rnE = rnE_R + 1j * rnE_I

            v.append(2 * np.real(Cn) + np.abs(Cn) * (np.exp(1j * Tn1) + np.exp(1j * Tn2)))
            v.append(rnE/(w1-pnE) - rnE.conj()/(w1+pnE.conj()))


    elif fctType == "szf":
        (H0, pnC_R, pnC_I, pnI_I, pnE_R, pnE_I, znC_R, znC_I, znI_I, znE_R, znE_I,
         poleOrigin, reversibleZeros) = parameterList[2:-1]

        v.append(H0)

        if tensorized:

            if poleOrigin:
                v.append(1/w1 + 0j)

            pnC = torch.abs(pnC_R) - 1j*torch.abs(pnC_I)
            znC = torch.abs(znC_R) + 1j*znC_I if not reversibleZeros else torch.abs(znC_R) - 1j*torch.abs(znC_I)
            v.append((w1-znC))
            v.append(1/(w1-pnC))
            v.append((w1+znC.conj()))
            v.append(1/(w1+pnC.conj()))

            pnI = -1j * torch.abs(pnI_I)
            znI = 1j * znI_I if (not reversibleZeros) else -1j * torch.abs(znI_I)
            v.append(w1-znI)
            v.append(1/(w1-pnI))

            pnE = torch.abs(pnE_R) - 1j*torch.abs(pnE_I) if stableEffectivePoles else torch.abs(pnE_R) + 1j*pnE_I
            znE = torch.abs(znE_R) - 1j*torch.abs(znE_I) if reversibleZeros else torch.abs(znE_R) + 1j*znE_I
            v.append((w1-znE))
            v.append(1/(w1-pnE))
            v.append((w1+znE.conj()))
            v.append(1/(w1+pnE.conj()))

        else:

            if poleOrigin:
                v.append(1/w1 + 0j)

            pnC = np.abs(pnC_R) - 1j*np.abs(pnC_I)
            znC = np.abs(znC_R) + 1j*znC_I if not reversibleZeros else np.abs(znC_R) - 1j*np.abs(znC_I)
            v.append((w1-znC))
            v.append(1/(w1-pnC))
            v.append((w1+znC.conj()))
            v.append(1/(w1+pnC.conj()))

            pnI = -1j * np.abs(pnI_I)
            znI = 1j * znI_I if (not reversibleZeros) else -1j * np.abs(znI_I)
            v.append(w1-znI)
            v.append(1/(w1-pnI))

            pnE = np.abs(pnE_R) - 1j*np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j*pnE_I
            znE = np.abs(znE_R) - 1j*np.abs(znE_I) if reversibleZeros else np.abs(znE_R) + 1j*znE_I
            v.append((w1-znE))
            v.append(1/(w1-pnE))
            v.append((w1+znE.conj()))
            v.append(1/(w1+pnE.conj()))

    else:
        print('Invalid fct type. Available options are "szf", "sem", "gdl" and "mtse" ')
        exit()

    return fctType, v



def H(fctType, H_terms, tensorized=True):
    v = 0.
    if fctType == "szf":
        v = 1.
        H0 = torch.real(H_terms[0]) if tensorized else np.real(H_terms[0])
        for resTerms in H_terms[1:]:
            if resTerms.ndim > 1:
                if tensorized:
                    v *= torch.prod(resTerms, dim=resTerms.ndim-1)
                else:
                    v *= np.prod(resTerms, axis=resTerms.ndim-1)
            else:
                v *= resTerms
        v *= H0
    else:
        Hnr = torch.real(H_terms[0]) if tensorized else np.real(H_terms[0])
        for resTerms in H_terms[1:]:
            if resTerms.ndim > 1:
                if tensorized:
                    v += resTerms.sum(dim=resTerms.ndim-1)
                else:
                    v += resTerms.sum(axis=resTerms.ndim-1)
            else:
                v += resTerms
        v += Hnr
    return v



def LossFct(Pr, Ta, alpha=(1,0,0,0), imaginaryPartPositive = False):

    l1 = torch.sum(torch.abs(Ta - Pr)**2) / torch.sum(torch.abs(Ta)**2) if alpha[0]>1e-5 else 0.
    l2 = torch.max(torch.abs(Ta - Pr) / (torch.abs(Ta) + 1e-3)) if alpha[1]>1e-5 else 0.

    dA = torch.abs(torch.real(Ta)) + 0.5 if alpha[2]>1e-5 else 0.
    A = torch.abs(torch.real(Ta-Pr)) if alpha[2]>1e-5 else 0.
    l3 = torch.mean(A/dA) if alpha[2]>1e-5 else 0.

    dB = torch.abs(torch.imag(Ta)) + 0.5 if alpha[3]>1e-5 else 0.
    B = torch.abs(torch.imag(Ta-Pr)) if alpha[3]>1e-5 else 0.
    l4 = torch.mean(B/dB) if alpha[3]>1e-5 else 0.

    l = alpha[0]*l1 + alpha[1]*l2 + alpha[2]*l3 + alpha[3]*l4

    if imaginaryPartPositive:
        l += 0.02*torch.relu( -torch.imag(Pr).sum() )

    return l




def FitModel(W, Ta, parameterList, imaginaryPartPositive=False, alpha=(1,0,0,0), tensorizeInput=False,
             nIter=2500, lr=3.3e-2, scIter=1200, scq=0.87,
             vizIteration=1000, visualizeLoss=True, loss_viz=None, logScalePoles=False):

    if tensorizeInput:
        W = torch.from_numpy(W)
        Ta = torch.from_numpy(Ta)

    fctType = parameterList[0]
    useZeros = True if fctType=="szf" else False
    scatterColorsZeros = None

    if useZeros:
        nmbPoles = 0
        poles, zeros = SPP.GetPolesAndZeros(parameterList)
        scatterColorsPoles = np.random.randint(0, 255, (poles.shape[0], 3))
        scatterColorsZeros = np.random.randint(0, 255, (zeros.shape[0], 3))

    else:
        nmbPoles = 0
        poles, residues = SPP.GetPolesAndResidues(parameterList)
        scatterColorsPoles = np.random.randint(0, 255, (poles.shape[0], 3))


    def lambda1(epoch):
        v = scq ** (epoch // scIter)
        # if lr * v < 5e-5:
        #     v = 5e-5 / lr
        return v

    optimizer = None
    if fctType == "sem":
        optimizer = optim.AdamW(parameterList[2:-2], lr=lr)
    elif fctType == "szf":
        optimizer = optim.AdamW(parameterList[2:-3], lr=lr)
    else:
        optimizer = optim.AdamW(parameterList[2:-1], lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    lossValues = np.zeros(nIter)
    bar = trange(nIter, desc="Fitting SEM")
    for i in bar:

        optimizer.zero_grad()

        fctType, resonantTerms = H_terms(W, parameterList)
        Pr = H(fctType, resonantTerms, tensorized=True)

        Ln = LossFct(Pr, Ta, alpha=alpha, imaginaryPartPositive=imaginaryPartPositive)
        lossValues[i] = Ln.detach().numpy()

        if not (torch.isnan(Ln)):

            Ln.backward()
            optimizer.step()
            scheduler.step()

            i += 1

            if visualizeLoss:
                with torch.no_grad():
                    if (i % vizIteration) == 0:
                        # LOSS FUNCTION =============================================================================
                        # ===========================================================================================
                        vizDict = dict(xlabel="iteration",
                                       ylabel="loss",
                                       title="Cost - " + str(scheduler.get_last_lr()),
                                       ytype="log")
                        loss_viz.line(X=np.arange(i),
                                      Y=lossValues[:i],
                                      opts=vizDict,
                                      win="cost")



                        # Real, Imag, Abs curves ====================================================================
                        # ===========================================================================================
                        Ta_n = Ta.detach().numpy()
                        Pr_n = Pr.detach().numpy()


                        y = np.vstack([np.abs(Ta_n), np.abs(Pr_n)])
                        x = np.vstack([W, W])
                        vizDict = dict(xlabel="w",
                                       title="|H(w)|",
                                       xtype="log",
                                       legend=["Target", "Reconstructed"])
                        loss_viz.line(X=x.transpose(),
                                      Y=y.transpose(),
                                      opts=vizDict,
                                      win="abs H")

                        y = np.vstack([np.real(Ta_n), np.real(Pr_n)])
                        x = np.vstack([W, W])
                        vizDict = dict(xlabel="w",
                                       title="Re[H(w)]",
                                       xtype="log",
                                       legend=["Target", "Reconstructed"])
                        loss_viz.line(X=x.transpose(),
                                      Y=y.transpose(),
                                      opts=vizDict,
                                      win="real H")

                        y = np.vstack([np.imag(Ta_n), np.imag(Pr_n)])
                        x = np.vstack([W, W])
                        vizDict = dict(xlabel="w",
                                       title="Im[H(w)]",
                                       xtype="log",
                                       legend=["Target", "Reconstructed"])
                        loss_viz.line(X=x.transpose(),
                                      Y=y.transpose(),
                                      opts=vizDict,
                                      win="imag H")




                        # Poles, residues, zeros ====================================================================
                        # ===========================================================================================
                        polesArray, residuesArray, zerosArray = np.array([]), np.array([]), np.array([])
                        if useZeros:
                            poles, zeros = SPP.GetPolesAndZeros(parameterList)
                            polesArray = poles + 1e-2
                            zerosArray = zeros + 1e-2
                        else:
                            poles, residues = SPP.GetPolesAndResidues(parameterList)
                            polesArray = poles + 1e-2
                            residuesArray = residues


                        xtype = "log" if logScalePoles else "linear"
                        x = np.vstack([np.real(polesArray), np.imag(polesArray)]).T
                        vizDict = dict(xlabel="R",
                                       ylabel="iR",
                                       title="poles",
                                       xtype=xtype,
                                       markercolor=scatterColorsPoles)
                        loss_viz.scatter(X=x.transpose(0, 1),
                                         opts=vizDict,
                                         win="poles")

                        if useZeros:
                            x = np.vstack([np.real(zerosArray), np.imag(zerosArray)]).T
                            vizDict = dict(xlabel="R",
                                           ylabel="iR",
                                           title="zeros",
                                           xtype=xtype,
                                           markercolor=scatterColorsZeros)
                            loss_viz.scatter(X=x.transpose(0, 1),
                                             opts=vizDict,
                                             win="zeros")
                        else:
                            x = np.vstack([np.real(residuesArray), np.imag(residuesArray)]).T
                            vizDict = dict(xlabel="R",
                                           ylabel="iR",
                                           title="residues",
                                           markercolor=scatterColorsPoles)
                            loss_viz.scatter(X=x.transpose(0, 1),
                                             opts=vizDict,
                                             win="residues")

        else:
            print("ERREUR")
            break

    return parameterList
