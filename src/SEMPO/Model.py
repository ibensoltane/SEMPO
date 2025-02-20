import numpy as np

c = 3e8
def G_ab(w, a, b):
    dP = a.shape[0] - 1
    dQ = b.shape[0] - 1
    w1 = w
    if np.isscalar(w):
        w1 = [w]
    va = np.polynomial.polynomial.polyvander(w1, dP)
    vb = np.polynomial.polynomial.polyvander(w1, dQ)
    p = (va * a).sum(axis=len(va.shape) - 1)
    q = (vb * b).sum(axis=len(vb.shape) - 1)

    return p / q


def G_SZF(w, g0, poles, zeros):
    w1 = w[..., None] if not np.isscalar(w) else np.array([w])[..., None]

    v = (g0 * np.prod(w1-zeros, axis=len(w1.shape)-1) /
         np.prod(w1-poles, axis=len(w1.shape)-1))
    return v


def H_SEM(w, H_NR, p, r):
    w1 = w[..., np.newaxis]
    v = H_NR + (r/(w1-p)).sum(axis=len(w1.shape)-1)
    return v


def H_HermitianSEM(w, H_NR, p, r):
    w1 = w[..., np.newaxis]
    v = H_NR + (r/(w1-p) - r.conj()/(w1+p.conj())).sum(axis=len(w1.shape)-1)
    # dK = 0
    # v = (-1)**dK * r/(w1 - p)**(dK+1) - (-1)**dK * r.conj()/(w1 + p.conj())**(dK+1)
    # v += H_NR + v.sum(axis=w.ndim)
    return v


def MergePoles(poles, residues, W, d0=0.5, nPts=120):
    p = poles.copy()
    r = residues.copy()
    wa = W[0]
    wb = W[-1]
    w = np.linspace(wa, wb, nPts)

    Hterms = (r/(w[..., None]-p) - r.conj()/(w[..., None]-p.conj()))
    keepMerging = True
    while keepMerging:
        D = np.ones((p.shape[0], p.shape[0])) * 1e10
        for a in range(p.shape[0] - 1):
            ra = r[a]
            Ra = np.abs(ra)
            pa = p[a]
            ha = Hterms[:, a]
            for b in range(a + 1, p.shape[0]):
                rb = r[b]
                Rb = np.abs(rb)
                pb = p[b]
                hb = Hterms[:, b]

                rc = ra + rb
                RC = Ra + Rb
                pc = 1 / RC * (Ra * pa + Rb * pb)
                hc = (rc / (w - pc) - rc.conj() / (w - pc.conj()))

                D[a, b] = np.abs(hc - (ha + hb)).sum() / (np.abs(ha + hb).sum()) * 100

        minDID = np.argmin(D)
        rID, cID = minDID // D.shape[1], minDID % D.shape[1]
        minD = D[rID, cID]
        if minD < d0:
            id1 = min(rID, cID)
            id2 = max(rID, cID)

            r1 = r[id1]
            R1 = np.abs(r1)
            p1 = p[id1]
            r2 = r[id2]
            R2 = np.abs(r2)
            p2 = p[id2]
            rc = r1 + r2
            RC = R1 + R2
            pc = 1 / RC * (R1 * p1 + R2 * p2)

            p = np.hstack([p[:id1], p[(id1 + 1):id2], p[(id2 + 1):], pc])
            r = np.hstack([r[:id1], r[(id1 + 1):id2], r[(id2 + 1):], rc[..., None]])
        else:
            keepMerging = False

    return p,r


def RemovePoles(hnr, poles, residues, W, d0=0.1, s0=1, nPts=120):
    p = poles.copy()
    r = residues.copy()
    wa = W[0]
    wb = W[-1]
    w = np.linspace(wa, wb, nPts)

    Hterms = (r / (w[..., None] - p) - r.conj() / (w[..., None] - p.conj()))
    Yw = hnr + Hterms.sum(axis=1)

    idsToRemove = []
    for a in range(p.shape[0] - 1):
        ra = r[a]
        pa = p[a]
        ha = Hterms[:, a]

        s = np.std(np.abs(ha)) / np.std(np.abs(Yw)) * 100
        if s < s0:
            idsToRemove.append(a)
            hnr += 2 * np.real(ra / pa)
        else:
            d = np.abs(ha - Yw).sum() / (np.abs(Yw).sum()) * 100
            if d > (100 - d0):
                idsToRemove.append(a)

    if len(idsToRemove) > 0:
        p = np.delete(p, idsToRemove)
        r = np.delete(r, idsToRemove)

    return hnr, p, r


def StabilizePoles(p, r, q0=1e-5):
    rC = []
    pC = []
    for i in range(p.shape[0]):
        pi = p[i]
        ri = r[i]
        if np.abs(np.imag(pi)) < 1e-10:
            pC.append(np.real(pi) - 1j * q0)
            rC.append(ri / 2)
            pC.append(np.real(pi) + 1j * q0)
            rC.append(ri.conj() / 2)
        else:
            pC.append(pi)
            rC.append(ri)
    pC = np.array(pC)
    rC = np.array(rC)

    return pC, rC