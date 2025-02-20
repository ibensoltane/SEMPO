import torch
import numpy as np

from . import Cauchy as SPC

c = 3e8
hb = 6.582119570e-1


# hb = 1


def Tensorize(parameterList):
    if not parameterList[1]:
        fctType = parameterList[0]

        paramRange = np.arange(2, len(parameterList)-1)
        if fctType=="sem":
            paramRange = np.arange(2, len(parameterList)-2)
        elif fctType=="szf":
            paramRange = np.arange(2, len(parameterList)-3)

        for i in paramRange:
            p = parameterList[i]
            if np.isscalar(p):
                parameterList[i] = torch.tensor(p)
            else:
                parameterList[i] = torch.from_numpy(p)
            parameterList[i].requires_grad_(True)

        parameterList[1] = True

    return parameterList



def Numpify(oldParameterList):
    parameterList = None
    if oldParameterList[1]:
        parameterList = [oldParameterList[0], oldParameterList[1]]
        fctType = oldParameterList[0]

        paramRange = np.arange(2, len(oldParameterList) - 1)
        if fctType == "sem":
            paramRange = np.arange(2, len(oldParameterList) - 2)
        elif fctType == "szf":
            paramRange = np.arange(2, len(oldParameterList) - 3)

        for i in paramRange:
            p = oldParameterList[i].clone()
            if torch.is_tensor(p):
                parameterList.append(p.detach().cpu().numpy())

        parameterList += oldParameterList[paramRange[-1]+1:]
        parameterList[1] = False

    else:
        parameterList = oldParameterList

    return parameterList



def GenerateSEMParameterList(fctType="sem", tensorized=True,
                             Hnr=None, residues=None, poles=None,
                             nPoles=0, nPolesIm=0, nPolesEff=0, stableEffectivePoles=False, poleOrigin=False,
                             w=None, minDist=2e-4):
    parameterList = None

    Hnr = np.random.randn(1) if (Hnr is None) else Hnr

    polesAdd = None
    residuesAdd = None
    if w is None:
        polesAdd = [np.abs(np.random.randn(nPoles) * 5) - 1j * np.abs(np.random.randn(nPoles) * 5)]
        polesAdd.append(-1j * np.abs(np.random.randn(nPolesIm)) * 0.001)
        if stableEffectivePoles:
            polesAdd.append(np.abs(np.random.randn(nPolesEff) * 5) - 1j * np.abs(np.random.randn(nPolesEff) * 5))
        else:
            polesAdd.append(np.abs(np.random.randn(nPolesEff) * 5) + 1j * np.random.randn(nPolesEff) * 5)
    else:
        wa = 0.7 * w[0]
        wb = 1.3 * w[-1]
        polesAdd = [np.abs(np.linspace(wa, wb, nPoles)) - 0.05j * np.abs(np.linspace(wa, wb, nPoles))]
        polesAdd.append(-0.01j * np.abs(np.linspace(wa, wb, nPolesIm)))
        if stableEffectivePoles:
            polesAdd.append(np.abs(np.linspace(1.2 * wb, 2 * wb, nPolesEff)) - 0.01j * np.abs(
                np.linspace(1.2 * wb, 2 * wb, nPolesEff)))
        else:
            polesAdd.append(
                np.abs(np.linspace(1.2 * wb, 2 * wb, nPolesEff)) + 0.01j * np.linspace(1.2 * wb, 2 * wb, nPolesEff))

    residuesAdd = [np.random.randn(nPoles) * 0.05 + 1j * np.random.randn(nPoles) * 0.05]
    residuesAdd.append(1j * np.random.randn(nPolesIm) * 5)
    residuesAdd.append(np.random.randn(nPolesEff) * 0.05 + 1j * np.random.randn(nPolesEff) * 0.05)

    if not poles is None:
        if residues is not None:
            residuesC = np.hstack([residues[np.real(poles) >= minDist], residuesAdd[0]])
            residuesI = np.hstack([residues[np.abs(np.real(poles)) < minDist], residuesAdd[1]])
            residuesEff = residuesAdd[2]
            residues = [residuesC, residuesI, residuesEff]

        polesC = np.hstack([poles[np.real(poles) >= minDist], polesAdd[0]])
        polesI = np.hstack([poles[np.abs(np.real(poles)) < minDist], polesAdd[1]])
        polesEff = polesAdd[2]
        poles = [polesC, polesI, polesEff]

        if residues is None:
            residues = [np.random.randn(polesC.shape[0]) * 5 + 1j * np.random.randn(polesC.shape[0]) * 5]
            residues.append(1j * np.random.randn(polesI.shape[0]) * 5)
            residues.append(np.random.randn(polesEff.shape[0]) * 5 + 1j * np.random.randn(polesEff.shape[0]) * 5)

    else:
        residues = [residuesAdd[0], residuesAdd[1], residuesAdd[2]]
        poles = [polesAdd[0], polesAdd[1], polesAdd[2]]

    if fctType == "sem":
        pnC, pnI, pnE = poles
        pnC_R, pnC_I = np.real(pnC), np.imag(pnC)
        pnI_I = np.imag(pnI)
        pnE_R, pnE_I = np.real(pnE), np.imag(pnE)

        rnC, rnI, rnE = residues
        r0 = 0.
        if poleOrigin:
            r0 = np.random.randn(1) * 5j
        rnC_R, rnC_I = np.real(rnC), np.imag(rnC)
        rnI_I = np.imag(rnI)
        rnE_R, rnE_I = np.real(rnE), np.imag(rnE)

        parameterList = [fctType, False, Hnr, pnC_R, pnC_I, rnC_R, rnC_I, pnI_I, rnI_I, pnE_R, pnE_I, rnE_R, rnE_I,
                         r0, poleOrigin, stableEffectivePoles]

    elif fctType == "gdl":
        pnC, pnI, pnE = poles
        pnC_R, pnC_I = np.real(pnC), np.imag(pnC)
        pnI_I = np.imag(pnI)
        pnE_R, pnE_I = np.real(pnE), np.imag(pnE)

        rnC, rnI, rnE = residues
        rnC_R, rnC_I = np.real(rnC), np.imag(rnC)
        rnE_R, rnE_I = np.real(rnE), np.imag(rnE)

        bn = -np.abs(pnI_I)
        an = np.sqrt(np.abs(bn * rnI))
        r0 = np.random.randn(1)
        G0 = r0 + ((an ** 2) / bn).sum()

        wn = np.abs(pnC_R)
        Gn = -2 * np.abs(pnC_I)
        s1n = -2 * rnC_I / Gn
        s2n = -2 * np.real(rnC * pnC.conj()) / (np.abs(pnC) ** 2)

        parameterList = [fctType, False, Hnr, G0, an, bn, s1n, Gn, wn, s2n, pnE_R, pnE_I, rnE_R, rnE_I,
                         stableEffectivePoles]

    elif fctType == "mtse":
        pnC, pnI, pnE = poles
        pnC_R, pnC_I = np.real(pnC), np.imag(pnC)
        pnI_I = np.imag(pnI)
        pnE_R, pnE_I = np.real(pnE), np.imag(pnE)

        rnC, rnI, rnE = residues
        rnE_R, rnE_I = np.real(rnE), np.imag(rnE)
        mrn = np.abs(rnC)
        trn = np.angle(rnC)

        parameterList = [fctType, False, Hnr, mrn, trn, pnC_R, pnC_I, pnE_R, pnE_I, rnE_R, rnE_I, stableEffectivePoles]

    else:
        print('Invalid fct type. Available options are "sem", "gdl" and "mtse" ')
        exit()

    if tensorized and (parameterList is not None):
        parameterList = Tensorize(parameterList)

    return parameterList


def GenerateSZFParameterList(tensorized=True, G0=None, poles=None, zeros=None,
                             nPoles=0, nPolesIm=0, nPolesEff=0, stableEffectivePoles=False, poleOrigin=False,
                             nZeros=0, nZerosIm=0, nZerosEff=0, reversibleZeros=False,
                             w=None, minDist=2e-4):
    parameterList = None

    G0 = np.random.randn(1) if (G0 is None) else G0

    polesAdd = None
    zerosAdd = None
    if w is None:
        polesAdd = [np.abs(np.random.randn(nPoles) * 5) - 1j * np.abs(np.random.randn(nPoles) * 5)]
        polesAdd.append(-1j * np.abs(np.random.randn(nPolesIm)) * 0.001)
        if stableEffectivePoles:
            polesAdd.append(np.abs(np.random.randn(nPolesEff) * 5) - 1j * np.abs(np.random.randn(nPolesEff) * 5))
        else:
            polesAdd.append(np.abs(np.random.randn(nPolesEff) * 5) + 1j * np.random.randn(nPolesEff) * 5)

        if reversibleZeros:
            zerosAdd = [np.abs(np.random.randn(nZeros) * 5) - 1j * np.abs(np.random.randn(nZeros) * 5)]
            zerosAdd.append(-1j * np.abs(np.random.randn(nZerosIm)) * 0.001)
            zerosAdd.append(np.abs(np.random.randn(nZerosEff) * 5) - 1j * np.abs(np.random.randn(nZerosEff)) * 5)
        else:
            zerosAdd = [np.abs(np.random.randn(nZeros) * 5) + 1j * np.random.randn(nZeros) * 5]
            zerosAdd.append(1j * np.random.randn(nZerosIm) * 0.001)
            zerosAdd.append(np.abs(np.random.randn(nZerosEff) * 5) + 1j * np.random.randn(nZerosEff) * 5)
    else:
        wa = 0.7 * w[0]
        wb = 1.3 * w[-1]
        polesAdd = [np.abs(np.linspace(wa, wb, nPoles)) - 0.1j * np.abs(np.linspace(wa, wb, nPoles))]
        polesAdd.append(-0.1j * np.abs(np.linspace(wa, wb, nPolesIm)))
        if stableEffectivePoles:
            polesAdd.append(np.abs(np.linspace(1.2 * wb, 2 * wb, nPolesEff)) - 0.1j * np.abs(
                np.linspace(1.2 * wb, 2 * wb, nPolesEff)))
        else:
            polesAdd.append(
                np.abs(np.linspace(1.2 * wb, 2 * wb, nPolesEff)) + 0.1j * np.linspace(1.2 * wb, 2 * wb, nPolesEff))

        if reversibleZeros:
            zerosAdd = [np.abs(np.linspace(wa, wb, nZeros)) - 0.2j * np.abs(np.linspace(wa, wb, nZeros))]
            zerosAdd.append(-0.2j * np.abs(np.linspace(wa, wb, nZerosIm)))
            zerosAdd.append(np.abs(np.linspace(1.2 * wb, 2 * wb, nZerosEff)) - 0.2j * np.abs(
                np.linspace(1.2 * wb, 2 * wb, nZerosEff)))
        else:
            zerosAdd = [np.abs(np.linspace(wa, wb, nZeros)) + 0.2j * np.linspace(wa, wb, nZeros)]
            zerosAdd.append(0.2j * np.linspace(wa, wb, nZerosIm))
            zerosAdd.append(
                np.abs(np.linspace(1.2 * wb, 2 * wb, nZerosEff)) + 0.2j * np.linspace(1.2 * wb, 2 * wb, nZerosEff))

    if not poles is None:
        polesC = np.hstack([poles[np.real(poles) >= minDist], polesAdd[0]])
        polesI = np.hstack([poles[np.abs(np.real(poles)) < minDist], polesAdd[1]])
        polesEff = polesAdd[2]
        poles = [polesC, polesI, polesEff]

    else:
        poles = [polesAdd[0], polesAdd[1], polesAdd[2]]

    if not zeros is None:
        zerosC = np.hstack([zeros[np.real(zeros) >= minDist], zerosAdd[0]])
        zerosI = np.hstack([zeros[np.abs(np.real(zeros)) < minDist], zerosAdd[1]])
        zerosEff = zerosAdd[2]
        zeros = [zerosC, zerosI, zerosEff]

    else:
        zeros = [zerosAdd[0], zerosAdd[1], zerosAdd[2]]

    pnC, pnI, pnE = poles
    pnC_R, pnC_I = np.real(pnC), np.imag(pnC)
    pnI_I = np.imag(pnI)
    pnE_R, pnE_I = np.real(pnE), np.imag(pnE)

    znC, znI, znE = zeros
    znC_R, znC_I = np.real(znC), np.imag(znC)
    znI_I = np.imag(znI)
    znE_R, znE_I = np.real(znE), np.imag(znE)

    parameterList = ["szf", False, G0, pnC_R, pnC_I, pnI_I, pnE_R, pnE_I, znC_R, znC_I, znI_I, znE_R, znE_I,
                     poleOrigin, reversibleZeros, stableEffectivePoles]

    if tensorized and (parameterList is not None):
        parameterList = Tensorize(parameterList)

    return parameterList


def ConvertToSEM(parameterList, wa=0.):
    newParameterList = None

    fctType = parameterList[0]
    tensorized = parameterList[1]
    stableEffectivePoles = parameterList[-1]

    if tensorized:
        parameterList = Numpify(parameterList)

    Hnr = parameterList[2]

    if fctType == "gdl":
        G0, an, bn, s1n, Gn, wn, s2n, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[3:-1]

        s1nn = s1n
        Gnn = 0.00005 + np.abs(Gn)
        wnn = 0.00005 + np.abs(wn)
        s2nn = s2n

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        pnC = wnn - 1j * Gnn / 2
        pnI = -1j * np.abs(bn)

        rnE = rnE_R + 1j * rnE_I
        rnC = -(1j * s1nn * Gnn * (wnn - 1j * (Gnn / 2)) + s2nn * (wnn ** 2 + (Gnn / 2) ** 2)) / (2 * wnn)
        rnI = -1j * an ** 2 / np.abs(bn)

        r0 = 1j * G0 - rnI.sum()
        poleOrigin = True

        newParameterList = ["sem", tensorized, Hnr,
                            np.real(pnC), np.imag(pnC), np.real(rnC), np.imag(rnC),
                            np.imag(pnI), np.imag(rnI),
                            np.real(pnE), np.real(pnE), np.real(rnE), np.real(rnE),
                            r0, poleOrigin, stableEffectivePoles]


    elif fctType == "mtse":
        mrn, trn, pnC_R, pnC_I, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[3:-1]

        rnC = np.abs(mrn) * np.exp(1j * trn)
        rnC_R, rnC_I = np.real(rnC), np.imag(rnC)
        r0 = 0.

        poleOrigin = False

        newParameterList = ["sem", tensorized, Hnr,
                            pnC_R, pnC_I, rnC_R, rnC_I,
                            np.array([]), np.array([]),
                            pnE_R, pnE_I, rnE_R, rnE_I,
                            r0, poleOrigin, stableEffectivePoles]


    elif fctType == "szf":
        (G0, pnC_R, pnC_I, pnI_I, pnE_R, pnE_I, znC_R, znC_I, znI_I, znE_R, znE_I,
         poleOrigin, reversibleZeros) = parameterList[2:-1]

        pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
        znC = np.abs(znC_R) + 1j * znC_I if not reversibleZeros else np.abs(znC_R) - 1j * np.abs(znC_I)

        pnI = -1j * np.abs(pnI_I)
        znI = 1j * znI_I / 2 if (not reversibleZeros) else -1j * np.abs(znI_I) / 2

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        znE = np.abs(znE_R) - 1j * np.abs(znE_I) if reversibleZeros else np.abs(znE_R) + 1j * znE_I

        poles = np.hstack([pnC, pnI, pnE])
        zeros = np.hstack([znC, znI, znE])

        residues = SPC.GetResidueFromCauchy(G0, poles, zeros)
        HNR = SPC.GetHnrFromCauchy(G0, poles, zeros, residues, wa=wa)

        newParameterList = GenerateSEMParameterList(fctType="sem", tensorized=tensorized,
                                                    Hnr=HNR, residues=residues, poles=poles,
                                                    nPoles=0, nPolesIm=0, nPolesEff=0,
                                                    stableEffectivePoles=stableEffectivePoles, poleOrigin=poleOrigin)


    else:
        newParameterList = parameterList

    if tensorized:
        newParameterList = Tensorize(newParameterList)

    return newParameterList


def ConvertToGDL(oldParameterList, wa=0.):
    newParameterList = None

    fctType = oldParameterList[0]
    tensorized = oldParameterList[1]
    stableEffectivePoles = oldParameterList[-1]

    parameterList = None
    if not (fctType == "sem" or fctType == "gdl"):
        parameterList = ConvertToSEM(oldParameterList, wa=wa)
    else:
        parameterList = oldParameterList

    if tensorized:
        parameterList = Numpify(parameterList)

    fctType = parameterList[0]
    if fctType == "sem":
        Hnr, pnC_R, pnC_I, rnC_R, rnC_I, pnI_I, rnI_I, pnE_R, pnE_I, rnE_R, rnE_I, r0, poleOrigin = parameterList[2:-1]

        pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
        rnC = rnC_R + 1j * rnC_I

        pnI = -1j * np.abs(pnI_I)
        rnI = 1j * rnI_I / 2

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        pnE_R, pnE_I = np.real(pnE), np.imag(pnE)
        rnE = rnE_R + 1j * rnE_I
        rnE_R, rnE_I = np.real(rnE), np.imag(rnE)

        bn = -np.abs(pnI)
        an = np.sqrt(np.abs(bn * rnI))
        r0 = np.random.randn(1)
        G0 = r0 + ((an ** 2) / bn).sum()

        wn = np.abs(pnC_R)
        Gn = -2 * np.abs(pnC_I)
        s1n = -2 * rnC_I / Gn
        s2n = -2 * np.real(rnC * pnC.conj()) / (np.abs(pnC) ** 2)

        newParameterList = ["gdl", tensorized, Hnr, G0, an, bn, s1n, Gn, wn, s2n,
                            pnE_R, pnE_I, rnE_R, rnE_I,
                            stableEffectivePoles]

    else:
        newParameterList = parameterList

    if tensorized:
        newParameterList[1] = False
        newParameterList = Tensorize(newParameterList)

    return newParameterList


def ConvertToMTSE(oldParameterList, wa=0.):
    newParameterList = None

    fctType = oldParameterList[0]
    tensorized = oldParameterList[1]
    stableEffectivePoles = oldParameterList[-1]

    parameterList = None
    if not (fctType == "sem" or fctType == "mtse"):
        parameterList = ConvertToSEM(oldParameterList, wa=wa)
    else:
        parameterList = oldParameterList

    if tensorized:
        parameterList = Numpify(parameterList)

    fctType = parameterList[0]
    if fctType == "sem":
        Hnr, pnC_R, pnC_I, rnC_R, rnC_I, pnI_I, rnI_I, pnE_R, pnE_I, rnE_R, rnE_I, r0, poleOrigin = parameterList[2:-1]

        pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
        pnI = -1j * np.abs(pnI_I)
        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I

        rnC = rnC_R + 1j * rnC_I
        rnI = 1j * rnI_I / 2
        rnE = rnE_R + 1j * rnE_I

        pnE = np.hstack([pnI, pnE])
        rnE = np.hstack([rnI, rnE])

        mrn = np.abs(rnC)
        trn = np.angle(rnC)

        newParameterList = ["mtse", tensorized, Hnr,
                            mrn, trn, np.real(pnC), np.imag(pnC),
                            np.real(pnE), np.imag(pnE), np.real(rnE), np.imag(rnE),
                            stableEffectivePoles]

    else:
        newParameterList = parameterList

    if tensorized:
        newParameterList[1] = False
        newParameterList = Tensorize(newParameterList)

    return newParameterList


def GetPolesAndResidues(oldParameterList, wa=0.):
    poles, residues = None, None

    fctType = oldParameterList[0]
    tensorized = oldParameterList[1]
    stableEffectivePoles = oldParameterList[-1]

    parameterList = None
    if fctType == "szf":
        parameterList = ConvertToSEM(oldParameterList, wa=wa)
    else:
        parameterList = oldParameterList if not tensorized else Numpify(oldParameterList)

    if tensorized:
        parameterList = Numpify(parameterList)

    fctType = oldParameterList[0]
    if fctType == "sem":
        Hnr, pnC_R, pnC_I, rnC_R, rnC_I, pnI_I, rnI_I, pnE_R, pnE_I, rnE_R, rnE_I, r0, poleOrigin = parameterList[2:-1]

        pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
        rnC = rnC_R + 1j * rnC_I

        pnI = -1j * np.abs(pnI_I)
        rnI = 1j * rnI_I / 2

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        rnE = rnE_R + 1j * rnE_I

        if poleOrigin:
            pnI = np.hstack([0., pnI])
            rnI = np.hstack([-1j * np.abs(r0) / 2, rnI])

        poles = np.hstack([pnC, pnI, pnE])
        residues = np.hstack([rnC, rnI, rnE])

    elif fctType == "gdl":
        Hnr, G0, an, bn, s1n, Gn, wn, s2n, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[2:-1]

        s1nn = s1n
        Gnn = 0.00005 + np.abs(Gn)
        wnn = 0.00005 + np.abs(wn)
        s2nn = s2n

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        rnE = rnE_R + 1j * rnE_I

        pC = wnn - 1j * Gnn / 2
        pI = np.hstack([0, -1j * np.abs(bn)])
        poles = np.hstack([pC, pI, pnE])

        resPoles = 1j * s1nn * Gnn * pC + s2nn * np.abs(pC) ** 2
        resPoles /= -2 * np.real(pC)
        resPolesIm = -1j * (an ** 2) / np.abs(bn)
        residues = np.hstack([resPoles, np.hstack([1j * G0 - resPolesIm.sum(), resPolesIm]) / 2, rnE])

    elif fctType == "mtse":
        Hnr, mrn, trn, wnR, wnI, pnE_R, pnE_I, rnE_R, rnE_I = parameterList[2:-1]

        rnC = np.abs(mrn) * np.exp(1j * trn)
        pnC = np.abs(wnR) - 1j * np.abs(wnI)

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        rnE = rnE_R + 1j * rnE_I

        poles = np.hstack([pnC, pnE])
        residues = np.hstack([rnC, rnE])

    else:
        print('Invalid fct type. Available options are "sem", "gdl" and "mtse" ')
        exit()

    return poles, residues


def GetPolesAndZeros(oldParameterList):
    poles, zeros = None, None

    fctType = oldParameterList[0]
    tensorized = oldParameterList[1]
    stableEffectivePoles = oldParameterList[-1]

    parameterList = oldParameterList if not tensorized else Numpify(oldParameterList)

    if fctType == "szf":
        (G0, pnC_R, pnC_I, pnI_I, pnE_R, pnE_I, znC_R, znC_I, znI_I, znE_R, znE_I,
         poleOrigin, reversibleZeros) = parameterList[2:-1]

        pnC = np.abs(pnC_R) - 1j * np.abs(pnC_I)
        znC = np.abs(znC_R) + 1j * znC_I if not reversibleZeros else np.abs(znC_R) - 1j * np.abs(znC_I)

        pnI = -1j * np.abs(pnI_I)
        znI = 1j * znI_I / 2 if (not reversibleZeros) else -1j * np.abs(znI_I) / 2

        if poleOrigin:
            pnI = np.hstack([0., pnI])

        pnE = np.abs(pnE_R) - 1j * np.abs(pnE_I) if stableEffectivePoles else np.abs(pnE_R) + 1j * pnE_I
        znE = np.abs(znE_R) - 1j * np.abs(znE_I) if reversibleZeros else np.abs(znE_R) + 1j * znE_I

        poles = np.hstack([pnC, pnI, pnE])
        zeros = np.hstack([znC, znI, znE])

    return poles, zeros