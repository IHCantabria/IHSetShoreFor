import numpy as np
from numba import njit
import math

@njit(fastmath=True, cache=True)
def shoreFor_Yini(P, Omega, dt_arr, phi, D, cp, cm, b, Sini):
    '''
    This function apply the ShoreFor (Davidson et al. 2013) model   
    '''
    dt = 0.0
    for j in range(dt_arr.shape[0]):
        dt += dt_arr[j]
    dt /= dt_arr.shape[0]

    tau = phi * 24.0
    alpha = math.exp(-math.log(10.0) * dt / tau)
    n = P.shape[0]
    OmegaEQ = np.empty(n)
    OmegaEQ[0] = Omega[0]
    for i in range(1, n):
        OmegaEQ[i] = alpha * OmegaEQ[i-1] + (1.0 - alpha) * Omega[i]

    # IDX = int(math.floor(D * 24.0 / dt))
    S = np.empty(n)
    diff_cm_cp = cm - cp

    # S0 = Sini


    # for i in range(IDX+1):
    #     S[i] = S0

    # for i in range(IDX+1, n):

    S[0] = Sini
    for i in range(1, n):
        sP = math.sqrt(P[i])
        F = sP * (OmegaEQ[i] - Omega[i])
        cond_neg = 1.0 if F < 0.0 else 0.0
        inc = F * (diff_cm_cp * cond_neg + cp) + b
        S[i] = S[i-1] + dt * inc

    return S, OmegaEQ

@njit(fastmath=True, cache=True)
def shoreFor_idx_Yini(P, Omega, dt_arr, phi, D, cp, cm, Sini, idx_arr):
    '''
    This function apply the ShoreFor (Davidson et al. 2013) model   
    '''
    dt = 0.0
    for j in range(dt_arr.shape[0]):
        dt += dt_arr[j]
    dt /= dt_arr.shape[0]

    tau = phi * 24.0
    alpha = math.exp(-math.log(10.0) * dt / tau)
    n = P.shape[0]
    OmegaEQ = np.empty(n)
    OmegaEQ[0] = Omega[0]
    for i in range(1, n):
        OmegaEQ[i] = alpha * OmegaEQ[i-1] + (1.0 - alpha) * Omega[i]

    IDX = int(math.floor(D * 24.0 / dt))
    S = np.empty(n)
    diff_cm_cp = cm - cp

    target = D * 24.0 / dt
    best = 0
    mind = abs(idx_arr[0] - target)
    for j in range(1, idx_arr.shape[0]):
        d = abs(idx_arr[j] - target)
        if d < mind:
            mind = d
            best = j
    S0 = Sini[best]

    for i in range(IDX+1):
        S[i] = S0

    for i in range(IDX+1, n):
        sP = math.sqrt(P[i])
        F = sP * (OmegaEQ[i] - Omega[i])
        cond_neg = 1.0 if F < 0.0 else 0.0
        inc = F * (diff_cm_cp * cond_neg + cp)
        S[i] = S[i-1] + dt * inc

    return S, OmegaEQ

# @njit(nopython=True, fastmath=True)
# def shoreFor(P, Omega, dt_arr, phi, D, cp, cm, Sini, idx_arr):
#     '''
#     This function apply the ShoreFor (Davidson et al. 2013) model   
#     '''
#     dt = np.mean(dt_arr)
#     tau = phi * 24.0
#     alpha = np.exp(-np.log(10.0) * dt / tau)
#     n = P.shape[0]
#     IDX = int(np.floor(D * 24.0 / dt))
    
#     OmegaEQ_full = np.empty(n)
#     OmegaEQ_full[0] = Omega[0]
#     for i in range(1, n):
#         OmegaEQ_full[i] = alpha * OmegaEQ_full[i-1] + (1.0 - alpha) * Omega[i]
    
#     F = np.empty(n)
#     for i in range(n):
#         F[i] = np.nan
#     for i in range(IDX, n):
#         F[i] = np.sqrt(P[i]) * (OmegaEQ_full[i] - Omega[i])
    
#     S = np.empty(n)
#     for i in range(n):
#         S[i] = np.nan
    
#     if Sini.size == 1:
#         S0 = Sini[0]
#     else:
#         target = D * 24.0 / dt
#         best = 0
#         mind = abs(idx_arr[0] - target)
#         for j in range(idx_arr.size):
#             d = abs(idx_arr[j] - target)
#             if d < mind:
#                 mind = d
#                 best = j
#         S0 = Sini[best]
#     S[IDX] = S0
    
#     for i in range(IDX+1, n):
#         inc = cm * (F[i] < 0) * F[i] + cp * (F[i] >= 0) * F[i]
#         S[i] = S[i-1] + dt * inc
    
#     for i in range(IDX):
#         S[i] = S0
    
#     return S, OmegaEQ_full