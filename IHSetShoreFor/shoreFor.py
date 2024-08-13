import numpy as np
from scipy.signal import convolve

def shoreFor(P, Omega, dt, phi=0, D=0, Sini=0, cp=0, cm=0):
    '''
    This function apply the ShoreFor (Davidson et al. 2013) model   
    '''
    
    ii = np.arange(1, (D*24)+1, dt)
    filter = 10 ** (-np.abs(ii) / (phi * 24))
    OmegaEQ = convolve(Omega, filter, mode='same')
    OmegaEQ = OmegaEQ / np.sum(filter)
    F = (P ** 0.5) * (OmegaEQ - Omega) / np.std(OmegaEQ)
    IDX = len(filter)
    F[:IDX - 1] = np.nan

    S = np.full(len(Omega), np.nan)

    rero = F < 0
    racr = F >= 0
    S[IDX] = Sini

    rero_F = cm * rero[IDX+1:] * F[IDX+1:]
    racr_F = cp * racr[IDX+1:] * F[IDX+1:]

    rero_F_prev = cm * rero[IDX:-1] * F[IDX:-1]
    racr_F_prev = cp * racr[IDX:-1] * F[IDX:-1]

    S[IDX+1:] = 0.5 * dt * np.cumsum(rero_F + racr_F + rero_F_prev + racr_F_prev) + S[IDX]
   

    return S, OmegaEQ