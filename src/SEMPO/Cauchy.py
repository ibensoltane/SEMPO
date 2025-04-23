import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

from . import Model as SPM



def OriginalCauchyMethod(H0, W0, nmbMaxPoles=40, dWorigin=1e-3, phys=True,
                         useLogPrec=False, logPrec=15, plotSingularValues=False, figID=11111):
    # Adjust some variables
    W1, H1 = None, None
    if not phys:
        W1, H1 = W0, H0
    else:
        ids1 = np.arange(0, W0.shape[0], 2)
        ids2 = np.arange(1, W0.shape[0], 2)
        W1 = np.hstack([-np.flip(W0[ids1]), W0[ids2]])
        H1 = np.hstack([np.flip(H0[ids1]).conj(), H0[ids2]])
    H = H1[..., np.newaxis]
    nmbMaxPoles = min(nmbMaxPoles, W0.shape[0])

    # Set dQ and dP
    Mp = nmbMaxPoles
    Mz = Mp - 1

    # First SVD to find the singular values
    Wd = np.vander(W1, N=Mp + 1, increasing=True)
    B = H * Wd
    A = np.vander(W1, N=Mz + 1, increasing=True)
    C = np.hstack([A, -B])
    U, S, Vh = np.linalg.svd(C)

    # Look at the log ratio of the singular values S and the max singular value, and the growth of S
    yS = np.log10(S / np.max(S))
    yDiffS = np.abs(yS[1:] - yS[:-1])

    # Define the max rank using either the LogPrec or the growth stop point
    rsMax = nmbMaxPoles
    if useLogPrec:
        rsMax = np.argwhere(yS > -logPrec)[-1, 0]
        rsMax = max(rsMax, 4)  # diffZP+3 = Mz:1 + Mp:1+diffZP + 2 - K:1
    else:
        rsMax = np.argwhere(yDiffS == 0)
        if len(rsMax) > 0:
            rsMax = rsMax[0, 0] + 1
        else:
            rsMax = np.argwhere(yS > -logPrec)[-1, 0] + 1
        rsMax = max(rsMax, 4)  # diffZP+3 = Mz:1 + Mp:1+diffZP + 2 - K:1

    # Plot the singular values if you want...
    if plotSingularValues:
        fig = plt.figure(figID, figsize=[5, 4])
        plt.clf()
        fig.subplots_adjust(left=.17, bottom=.15, right=.85)
        ax = fig.subplots(1)
        ax.plot(np.arange(S.shape[0]) + 1, yS, color="black", linewidth=2.4)
        ax.scatter(x=np.arange(S.shape[0]) + 1, y=yS, s=13, color="blue", marker="x")
        ax.set_xlabel("Index $i$ of singular value", fontsize=12)
        ax.set_ylabel(r"log[$\sigma_i / \sigma_{max}$]", fontsize=12)
        ax.axvline(rsMax, color="black", linestyle="--")
        ax2 = ax.twinx()
        ax2.plot(np.arange(S.shape[0] - 1) + 1, yDiffS, linestyle="--", color="red")
        ax2.set_ylabel("Derivative approximation", color="red", fontsize=12)
        fig.show()

    g0, p, z = None, None, None
    if rsMax >= 4:
        Mp = rsMax // 2
        Mz = Mp - 1
        if not (Mp + Mz + 1 == rsMax):
            Mp = (rsMax + 1) // 2
            Mz = Mp - 1
        A = np.vander(W1, N=Mz + 1, increasing=True)
        Wd = np.vander(W1, N=Mp + 1, increasing=True)
        B = H * Wd
        C = np.hstack([A, -B])

        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        S[-1] = 0
        C = np.dot(U * S, Vh)
        A = C[:, :(Mz + 1)]
        B = -C[:, (Mz + 1):]

        Q, R1 = np.linalg.qr(A, mode="complete")
        R11 = R1[:(Mz + 1), :(Mz + 1)]

        R2 = -np.matmul(Q.T, B)
        R21 = R2[:(Mz + 1), :]
        R22 = R2[(Mz + 1):, :]

        U, S, V = np.linalg.svd(R22, full_matrices=True)
        b = V.conj().T[:, -1]
        a = -np.matmul(np.linalg.inv(R11), np.matmul(R21, b))

        g0 = np.abs(a[-1] / b[-1]) if phys else a[-1] / b[-1]
        pa = np.polynomial.Polynomial(a)
        pb = np.polynomial.Polynomial(b)
        z = pa.roots()
        p = pb.roots()

        if (z.shape[0] > 0) and (p.shape[0] > 0):
            zIm = z[np.abs(np.real(z)) < dWorigin]
            zPairs = z[np.abs(np.real(z)) >= dWorigin]
            if zPairs.shape[0] > 0:
                zPairs = zPairs[np.real(zPairs) > -dWorigin]

            pIm = p[np.abs(np.real(p)) < dWorigin]
            pPairs = p[np.abs(np.real(p)) >= dWorigin]
            if pPairs.shape[0] > 0:
                pPairs = pPairs[np.real(pPairs) > -dWorigin]

            zeros = np.hstack([-np.flip(zPairs).conj(), 1j * np.imag(zIm), zPairs]) if phys else z
            poles = np.hstack([-np.flip(pPairs).conj(), 1j * np.imag(pIm), pPairs]) if phys else p
            Hwpred = SPM.G_SZF(W0, g0, poles, zeros)
            p, z = poles, zeros

    return g0, p, z


def CauchyMethodOptZP(H0, W0, nmbMaxPoles=40, dWorigin=1e-3, phys=True,
                      stability=False, qStability=.1,
                      useLogPrec=False, logPrec=15, diffZPMax=5, plotSingularValues=False, figID=11111):
    # Adjust some variables
    W1, H1 = None, None
    if not phys:
        W1, H1 = W0, H0
    else:
        # ids1 = np.arange(0, W0.shape[0], 2)
        # ids2 = np.arange(1, W0.shape[0], 2)
        # W1 = np.hstack([-np.flip(W0[ids1]), W0[ids2]])
        # H1 = np.hstack([np.flip(H0[ids1]).conj(), H0[ids2]])
        W1 = np.hstack([-np.flip(W0), W0])
        H1 = np.hstack([np.flip(H0).conj(), H0])
    H = H1[..., np.newaxis]
    nmbMaxPoles = min(nmbMaxPoles, W0.shape[0])
    e0 = (np.abs(H0) ** 2).mean()

    # Set dQ and dP
    Mp = max(nmbMaxPoles, diffZPMax + 1)
    Mz = Mp

    # First SVD to find the singular values
    Wd = np.vander(W1, N=Mp + 1, increasing=True)
    B = H * Wd
    A = np.vander(W1, N=Mz + 1, increasing=True)
    C = np.hstack([A, -B])
    U, S, Vh = np.linalg.svd(C)

    # Look at the log ratio of the singular values S and the max singular value, and the growth of S
    yS = np.log10(S / np.max(S))
    yDiffS = np.abs(yS[1:] - yS[:-1])

    # Define the max rank using either the LogPrec or the growth stop point
    rsMax = nmbMaxPoles
    if useLogPrec:
        rsMax = np.argwhere(yS > -logPrec)[-1, 0]
        rsMax = max(rsMax, diffZPMax + 3)  # diffZP+3 = Mz:1 + Mp:1+diffZP + 2 - K:1
    else:
        rsMax = np.argwhere(yDiffS == 0)
        if len(rsMax) > 0:
            rsMax = rsMax[0, 0] + 1
        else:
            rsMax = np.argwhere(yS > -logPrec)[-1, 0] + 1
        rsMax = max(rsMax, diffZPMax + 3)  # diffZP+3 = Mz:1 + Mp:1+diffZP + 2 - K:1

    # Plot the singular values if you want...
    if plotSingularValues:
        fig = plt.figure(figID, figsize=[5, 4])
        plt.clf()
        fig.subplots_adjust(left=.16, bottom=.15, right=.85)
        ax = fig.subplots(1)
        ax.plot(np.arange(S.shape[0]) + 1, yS, color="black")
        ax.scatter(x=np.arange(S.shape[0]) + 1, y=yS, s=7, color="blue", marker="x")
        ax.set_xlabel("Index $i$ of singular value")
        ax.set_ylabel(r"log[$\sigma_i / \sigma_{max}$]")
        ax.axvline(rsMax, color="black", linestyle="--")
        ax2 = ax.twinx()
        ax2.plot(np.arange(S.shape[0] - 1) + 1, yDiffS, linestyle="--", color="red")
        ax2.set_ylabel("Derivative approximation", color="red")
        fig.show()

    # Now start from that max rank and add poles and zeros until the error is low
    g0n = []
    zn = []
    pn = []
    error = []
    Mzn = []
    Mpn = []

    rsMax = max(3, rsMax)
    if not (rsMax % 2):
        rsMax += 1
    Mp0 = rsMax // 2
    Mz0 = Mp0
    A0 = np.vander(W1, N=Mz0 + 1, increasing=True)
    Wd = np.vander(W1, N=Mp0 + 1, increasing=True)
    B0 = H * Wd
    C0 = np.hstack([A0, -B0])

    for Mp in range(1, Mp0 + 1):
        Mzmin = max(Mp - diffZPMax, 1)
        Mzmax = min(Mp, Mz0 + 1)
        for Mz in range(Mzmin, Mzmax):
            Mzn.append(Mz)
            Mpn.append(Mp)

            g0 = None
            zeros = None
            poles = None
            Hwpred = None
            e = np.inf

            dimC = np.hstack([np.arange(Mz + 1), np.arange(Mz0 + 1, Mz0 + 1 + Mp + 1)])
            C = C0[:, dimC]

            # print(C.shape)
            U, S, Vh = np.linalg.svd(C, full_matrices=False)
            S[-1] = 0
            C = np.dot(U * S, Vh)
            A = C[:, :(Mz + 1)]
            B = -C[:, (Mz + 1):]
            # print(C.shape)

            Q, R1 = np.linalg.qr(A, mode="complete")
            R11 = R1[:(Mz + 1), :(Mz + 1)]
            #             print(Mz, Mp,  R1.shape, R11.shape)
            #             print()

            R2 = -np.matmul(Q.T, B)
            R21 = R2[:(Mz + 1), :]
            R22 = R2[(Mz + 1):, :]

            U, S, V = np.linalg.svd(R22, full_matrices=True)
            b = V.conj().T[:, -1]
            a = np.matmul(R21, b)
            a = -np.matmul(np.linalg.inv(R11), a)

            g0 = np.abs(a[-1] / b[-1]) if phys else a[-1] / b[-1]
            pa = np.polynomial.Polynomial(a)
            pb = np.polynomial.Polynomial(b)
            z = pa.roots()
            p = pb.roots()

            if (z.shape[0] > 0) and (p.shape[0] > 0):
                zIm = z[np.abs(np.real(z)) < dWorigin]
                zPairs = z[np.abs(np.real(z)) >= dWorigin]
                if zPairs.shape[0] > 0:
                    zPairs = zPairs[np.real(zPairs) > -dWorigin]

                pIm = p[np.abs(np.real(p)) < dWorigin]
                pPairs = p[np.abs(np.real(p)) >= dWorigin]
                if pPairs.shape[0] > 0:
                    pPairs = pPairs[np.real(pPairs) > -dWorigin]

                zeros = np.hstack([-np.flip(zPairs).conj(), 1j * np.imag(zIm), zPairs]) if phys else z
                poles = np.hstack([-np.flip(pPairs).conj(), 1j * np.imag(pIm), pPairs]) if phys else p
                Hwpred = SPM.G_SZF(W0, g0, poles, zeros)
                p, z = poles, zeros
                e = np.abs((Hwpred - H0) ** 2).mean() / e0 * 100
                if stability:
                    e += qStability * (np.imag(p) > 0).sum()

            error.append(e)
            g0n.append(g0)
            zn.append(z)
            pn.append(p)

    idMin = np.argmin(error)
    g0 = g0n[idMin]
    zeros = zn[idMin]
    poles = pn[idMin]

    return g0, poles, zeros


def ChainCauchyMethod(H0, W0, nmbWindows=10, nmbMaxPoles=40, dWorigin=1e-4, phys=True, stability=False,
                      qStability=.1, useLogPrec=False, logPrec=15, diffZPMax=5, splitEvenly=True):
    H0n = []
    W0n = []

    if splitEvenly:
        wSplit = np.array_split(W0, nmbWindows)
        for i in range(nmbWindows):
            w1 = wSplit[i][0]
            w2 = wSplit[i][-1]
            W0n.append(W0[(w1 <= W0) & (W0 <= w2)])
            H0n.append(H0[(w1 <= W0) & (W0 <= w2)])
    else:
        dW = (W0[-1] - W0[0]) / nmbWindows
        W0n = []
        H0n = []
        for i in range(nmbWindows):
            w1 = W0[0] + i * dW
            w2 = W0[0] + (i + 1) * dW if i < (nmbWindows - 1) else W0[-1]
            W0n.append(W0[(w1 <= W0) & (W0 <= w2)])
            H0n.append(H0[(w1 <= W0) & (W0 <= w2)])

    g0n = []
    pn = []
    zn = []
    for i in range(nmbWindows):
        W1 = W0n[i]
        H1 = H0n[i]
        g0, poles, zeros = CauchyMethodOptZP(H1, W1, nmbMaxPoles=nmbMaxPoles, dWorigin=dWorigin,
                                             diffZPMax=diffZPMax, phys=phys,
                                             stability=stability, qStability=qStability,
                                             useLogPrec=useLogPrec, logPrec=logPrec, plotSingularValues=False)
        g0n.append(g0)
        pn.append(poles)
        zn.append(zeros)

    return g0n, pn, zn, W0n, H0n


def StackChainResult(g0n, pn, zn, distMax=30.):
    nmbWindows = len(g0n)

    g0 = np.mean(g0n)
    p = np.array([])
    z = np.array([])
    for i in range(nmbWindows):
        pi = pn[i]
        zi = zn[i]
        p = np.hstack([p, pi])
        z = np.hstack([z, zi])

    p = p[np.abs(p) < distMax]
    z = z[np.abs(z) < distMax]

    # p = p[np.real(p) >= 0]
    # z = z[np.real(z) >= 0]

    return g0, p, z


def HardMergeChain(g0n, pn, zn, wn):
    hnr, poles, residues = 0., np.array([]), np.array([])
    for i in range(len(g0n)):
        w = wn[i]
        g0 = g0n[i]
        p = pn[i]
        z = zn[i]
        r = GetResidueFromCauchy(g0, p, z)
        h0 = GetHnrFromCauchy(g0, p, z, r, .2j)

        if i == 0:
            criterion = (np.real(p) <= w[-1])
            invC = np.logical_not(criterion)
            rC = r[criterion]
            pC = p[criterion]
            h0 += (r[invC] / p[invC]).sum()
        elif 0 < i < (len(g0n) - 1):
            criterion = (w[0] <= np.real(p)) & (np.real(p) <= w[-1])
            invC = np.logical_not(criterion)
            rC = r[criterion]
            pC = p[criterion]
            h0 += (r[invC] / p[invC]).sum()
        else:
            criterion = (w[0] <= np.real(p))
            invC = np.logical_not(criterion)
            rC = r[criterion]
            pC = p[criterion]
            h0 += (r[invC] / p[invC]).sum()

        hnr += h0
        poles = np.hstack([poles, pC])
        residues = np.hstack([residues, rC])

    return hnr / len(wn), poles, residues


def GetResidueFromCauchy(g0, poles, zeros):
    rp = np.zeros_like(poles) + 0j
    for i in range(poles.shape[0]):
        rp[i] = SPM.G_SZF(poles[i], g0, np.hstack([poles[:i], poles[(i + 1):]]), zeros)

    return rp


def GetHnrFromCauchy(g0, poles, zeros, residues, wa=0.):
    HNR = SPM.G_SZF(wa, g0, poles, zeros)
    HNR -= (residues / (wa - poles + 1e-10)).sum()

    return HNR