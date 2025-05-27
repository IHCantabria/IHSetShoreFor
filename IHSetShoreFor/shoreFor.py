import numpy as np
from numba import njit

@njit(nopython=True, fastmath=True)
def shoreFor(P, Omega, dt_arr, phi, D, cp, cm, Sini, idx_arr):
    '''
    This function apply the ShoreFor (Davidson et al. 2013) model   
    '''
    dt = np.mean(dt_arr)
    tau = phi * 24.0
    alpha = np.exp(-np.log(10.0) * dt / tau)
    n = P.shape[0]
    IDX = int(np.floor(D * 24.0 / dt))
    
    OmegaEQ_full = np.empty(n)
    OmegaEQ_full[0] = Omega[0]
    for i in range(1, n):
        OmegaEQ_full[i] = alpha * OmegaEQ_full[i-1] + (1.0 - alpha) * Omega[i]
    
    F = np.empty(n)
    for i in range(n):
        F[i] = np.nan
    for i in range(IDX, n):
        F[i] = np.sqrt(P[i]) * (OmegaEQ_full[i] - Omega[i])
    
    S = np.empty(n)
    for i in range(n):
        S[i] = np.nan
    
    if Sini.size == 1:
        S0 = Sini[0]
    else:
        target = D * 24.0 / dt
        best = 0
        mind = abs(idx_arr[0] - target)
        for j in range(idx_arr.size):
            d = abs(idx_arr[j] - target)
            if d < mind:
                mind = d
                best = j
        S0 = Sini[best]
    S[IDX] = S0
    
    for i in range(IDX+1, n):
        inc = cm * (F[i] < 0) * F[i] + cp * (F[i] >= 0) * F[i]
        S[i] = S[i-1] + dt * inc
    
    for i in range(IDX):
        S[i] = S0
    
    return S, OmegaEQ_full