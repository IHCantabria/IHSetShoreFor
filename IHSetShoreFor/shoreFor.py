import numpy as np
from scipy.signal import convolve

def shoreFor(P, Omega, dt, phi=0, D=0, Sini=0, cp=0, cm=0):
    '''
    This function calculates the equilibrium profile and the profile evolution   
    '''
    
    ii = np.arange(0, (D-1)*24, dt)
    phivecP = 10 ** (-np.abs(ii) / (phi * 24))
    IDX = len(phivecP)

    phivecP = np.hstack((np.zeros(IDX), phivecP))

    filter_c = np.flip(phivecP / np.sum(phivecP))

    OmegaEQ = convolve(Omega - np.mean(Omega), filter_c, mode='same') + np.mean(Omega)

    F = (P ** 0.5) * (OmegaEQ - Omega) / np.std(OmegaEQ)

    # F[:IDX - 1] = 0

    S = np.full(len(Omega), np.nan)

    rero = F < 0
    racr = F >= 0
    S[0] = Sini

    # if flag_r == 0:
        
    #     # filter_r = np.hstack((np.zeros(int(N*24/dt)), np.ones(int(N*24/dt))))
    #     # filter_r = np.ones(int(N*24/dt))
    #     # F_plus = convolve(F * racr, filter_r, mode='same')
    #     # F_minus = convolve(F * rero, filter_r, mode='same')
    #     F_plus = np.sum(F * racr)
    #     F_minus = np.sum(F * rero)
    #     r = np.absolute(F_plus / F_minus)
    #     # r_rero_F = r[1:] * rero[1:] * F[1:]
    #     r_rero_F = r * rero[1:] * F[1:]
    #     racr_F = racr[1:] * F[1:]
    #     # r_rero_F_prev = r[:-1] * rero[:-1] * F[:-1]
    #     r_rero_F_prev = r * rero[:-1] * F[:-1]
    #     racr_F_prev = racr[:-1] * F[:-1]
    #     S[1:] = 0.5 * dt * c * np.cumsum(r_rero_F + racr_F + r_rero_F_prev + racr_F_prev) + S[0]
    # else:
    rero_F = cm * rero[1:] * F[1:]
    racr_F = cp * racr[1:] * F[1:]

    rero_F_prev = cm * rero[:-1] * F[:-1]
    racr_F_prev = cp * racr[:-1] * F[:-1]

    S[1:] = 0.5 * dt * np.cumsum(rero_F + racr_F + rero_F_prev + racr_F_prev) + S[0]
   

    return S, OmegaEQ