import numpy as np
from scipy.signal import convolve

def shoreFor(P, Omega, dt, phi, D, cp, cm, Sini, idx = None):
    '''
    This function apply the ShoreFor (Davidson et al. 2013) model   
    '''

    ii = np.arange(0, ((D*24)/dt), dt)
    filter = 10 ** (-ii / (phi * 24))
    IDX = len(filter) - 1

    OmegaEQ = convolve(Omega, filter, mode='valid')
    OmegaEQ = OmegaEQ / np.sum(filter)
    F = np.full(len(P), np.nan)
    F[IDX:] = (P[IDX:] ** 0.5) * (OmegaEQ - Omega[IDX:]) # / np.std(OmegaEQ)

    S = np.full(len(P), np.nan)

    rero = F < 0
    racr = F >= 0

    if len(Sini) == 1:
        S[IDX] = Sini
    else:
        minidx = np.argmin(np.abs(idx - len(ii)))
        S[IDX] = Sini[minidx]

    rero_F = cm * rero[IDX+1:] * F[IDX+1:]
    racr_F = cp * racr[IDX+1:] * F[IDX+1:]

    rero_F_prev = cm * rero[IDX:-1] * F[IDX:-1]
    racr_F_prev = cp * racr[IDX:-1] * F[IDX:-1]

    S[IDX+1:] = dt * np.cumsum(rero_F + racr_F) + S[IDX] #☻ + rero_F_prev + racr_F_prev
   
    S[0:IDX-1] = S[IDX]

    return S, OmegaEQ