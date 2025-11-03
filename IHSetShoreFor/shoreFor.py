import numpy as np
from numba import njit
import math

# @njit(fastmath=True, cache=True)
# def shoreFor_Yini(P, Omega, dt_arr, phi, D, cp, cm, b, Sini):
#     '''
#     This function apply the ShoreFor (Davidson et al. 2013) model   
#     '''
#     dt = 0.0
#     for j in range(dt_arr.shape[0]):
#         dt += dt_arr[j]
#     dt /= dt_arr.shape[0]

#     tau = phi * 24.0
#     alpha = math.exp(-math.log(10.0) * dt / tau)
#     n = P.shape[0]
#     OmegaEQ = np.empty(n)
#     OmegaEQ[0] = Omega[0]
#     for i in range(1, n):
#         OmegaEQ[i] = alpha * OmegaEQ[i-1] + (1.0 - alpha) * Omega[i]

#     # IDX = int(math.floor(D * 24.0 / dt))
#     S = np.empty(n)
#     diff_cm_cp = cm - cp

#     # S0 = Sini


#     # for i in range(IDX+1):
#     #     S[i] = S0

#     # for i in range(IDX+1, n):

#     S[0] = Sini
#     for i in range(1, n):
#         sP = math.sqrt(P[i])
#         F = sP * (OmegaEQ[i] - Omega[i]) / np.std(OmegaEQ)
#         cond_neg = 1.0 if F < 0.0 else 0.0
#         inc = F * (diff_cm_cp * cond_neg + cp) + b
#         S[i] = S[i-1] + dt * inc

#     return S, OmegaEQ


@njit(fastmath=True, cache=True)
def shoreFor_Yini(P, Omega, dt_arr, phi_days, D_days, cp, cm, b, Sini):
    """
    ShoreFor (Davidson et al., 2013) without convolution:
        dx/dt = b + c * sqrt(P) * (Omega_eq - Omega)
    with asymmetric c: cp (accretion, Omega_eq > Omega), cm (erosion, Omega_eq < Omega),
    and finite-window (length D) base-10 exponential weighting for Omega_eq.

    Parameters
    ----------
    P : 1D array
        Deepwater wave power time series (W/m). Must be >= 0.
    Omega : 1D array
        Dimensionless fall velocity time series.
    dt_arr : 1D array
        Time-step series in HOURS. (We use its mean; paper uses Δt in days.)
    phi_days : float
        Memory decay parameter φ (days). Weight is 10^{-Δt/φ}.
    D_days : float
        Window length D (days). Paper typically uses D = 2*φ.
    cp : float
        Accretion rate coefficient (m s^-1 (W/m)^-0.5) when Omega_eq > Omega.
    cm : float
        Erosion rate coefficient (m s^-1 (W/m)^-0.5) when Omega_eq < Omega.
    b : float
        Trend term (m/s), independent of wave forcing.
    Sini : float
        Initial shoreline position at t=0 (m).

    Returns
    -------
    S : 1D array
        Shoreline position time series (m), same length as inputs.
    Omega_eq : 1D array
        Time-varying equilibrium Omega computed with finite-window exponential weights.
    """
    n = P.shape[0]
    # Mean dt in HOURS -> convert to DAYS for the base-10 decay
    dt_h = 0.0
    for j in range(dt_arr.shape[0]):
        dt_h += dt_arr[j]
    dt_h /= dt_arr.shape[0]
    dt_d = dt_h / 24.0

    # Base-10 decay per step: beta = 10^{-Δt/φ}
    # Use τ = φ days directly (no 24 factor here because dt_d is in days)
    if phi_days <= 0.0:
        phi_days = 1e-6  # safety
    beta = math.pow(10.0, -dt_d / phi_days)

    # Finite window length K ≈ D / Δt (in steps). At least 1.
    if D_days <= 0.0:
        D_days = 2.0 * phi_days
    K = int(math.floor(D_days / dt_d))
    if K < 1:
        K = 1

    # Precompute beta^K once
    betaK = math.pow(beta, K)

    # Finite-window exponentially weighted mean via rolling numerator/denominator:
    # N_t = beta*N_{t-1} + Omega_t           (t < K)
    #     = beta*N_{t-1} + Omega_t - beta^K * Omega_{t-K}  (t >= K)
    # D_t = beta*D_{t-1} + 1                 (t < K)
    #     = beta*D_{t-1} + 1 - beta^K        (t >= K)
    Omega_eq = np.empty(n, dtype=np.float64)
    N_num = Omega[0]
    D_den = 1.0
    Omega_eq[0] = N_num / D_den

    for t in range(1, n):
        if t < K:
            N_num = beta * N_num + Omega[t]
            D_den = beta * D_den + 1.0
        else:
            # subtract the contribution leaving the finite window (weight beta^K)
            N_num = beta * N_num + Omega[t] - betaK * Omega[t - K]
            D_den = beta * D_den + 1.0 - betaK

        # Guard denominator (shouldn’t be <=0 unless pathological inputs)
        if D_den <= 1e-12:
            D_den = 1e-12
        Omega_eq[t] = N_num / D_den

    # Time integration of S (explicit Euler)
    S = np.empty(n, dtype=np.float64)
    S[0] = Sini
    diff_cm_cp = cm - cp

    for t in range(1, n):
        # Ensure non-negative power
        P_t = P[t]
        if P_t < 0.0:
            P_t = 0.0

        # Forcing term F = sqrt(P) * (Omega_eq - Omega)
        F = math.sqrt(P_t) * (Omega_eq[t] - Omega[t])

        # Choose c- for erosion (F<0), c+ for accretion (F>=0) without branches
        # inc = F*cm if F<0 else F*cp  ->  inc = F * ( (cm-cp)*(F<0) + cp )
        cond_neg = 1.0 if F < 0.0 else 0.0
        c_eff = diff_cm_cp * cond_neg + cp

        # dx/dt = b + c_eff * F
        dSdt = b + c_eff * F
        S[t] = S[t-1] + dt_h * dSdt  # dt_h is hours -> seconds
    return S, Omega_eq

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