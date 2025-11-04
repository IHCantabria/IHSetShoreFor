import numpy as np
from typing import Any
from IHSetUtils.CoastlineModel import CoastlineModel
from IHSetUtils import wMOORE
from .shoreFor import shoreFor_Yini

class assimilate_ShoreFor(CoastlineModel):
    """
    ShoreFor (Davidson et al., 2013) - EnKF parameter assimilation.
    Keeps switch_D (diagnostic depth) but fixes Yini to first observation.
    Ensemble param vector (transformed space):
      switch_D==0: [phi, log(ca), log(ce), b]
      switch_D==1: [phi, log(ca), log(ce), b, D]
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='ShoreFor (Davidson et al., 2013)',
            mode='assimilation',
            model_type='CS',
            model_key='ShoreFor'
        )
        self.setup_forcing()

    # ----------------------
    # Forcing & pre-processing
    # ----------------------
    def setup_forcing(self):
        cfg = self.cfg
        self.switch_D = int(cfg['switch_D'])
        self.D50 = float(cfg['D50'])

        # Consistent floors (as in your calibration class)
        # Note: self.hb/self.depthb are set in _break_waves_snell() (already called by base __init__)
        # If breaking not requested, CoastlineModel sets hb=hs and depthb=hb/0.55
        self.tp[self.tp < 2.0] = 2.0
        self.hb[self.hb < 0.1] = 0.1
        self.depthb[self.depthb < 0.2] = 0.2

        # Segment (calibration-time) counterparts for assimilation mode
        # Base class created hs_s, tp_s, idx_calibration, etc., but did not create hb_s/tp_s for mode='assimilation'.
        # Build them consistently from full-length hb/tp using the calibration indices (jj).
        jj = self.idx_calibration
        self.tp_s = self.tp[jj]
        self.hb_s = self.hb[jj]

        # Wave power proxy and mobility parameter
        self.P   = self.hb**2 * self.tp
        self.P_s = self.hb_s**2 * self.tp_s
        self.ws = wMOORE(self.D50)
        self.Omega   = self.hb   / (self.ws * self.tp)
        self.Omega_s = self.hb_s / (self.ws * self.tp_s)

        # Initial shoreline: first obs in the split period
        self.Yini = float(self.Obs_splited[0])

    # ----------------------
    # Ensemble init in transformed space
    # ----------------------
    def init_par(self, population_size: int):
        # bounds expected in cfg:
        # lb = [phi_min, ca_min, ce_min, D_min] (if switch_D==1) ; b bounds are fixed here as [-1, 1] like your calibration
        # ub = [phi_max, ca_max, ce_max, D_max]
        # We sample log(ca), log(ce) to keep positivity inside model
        phi_lo, ca_lo, ce_lo, b_lo = self.lb[0], self.lb[1], self.lb[2], self.lb[3]
        phi_up, ca_up, ce_up, b_up = self.ub[0], self.ub[1], self.ub[2], self.ub[3]

        if self.switch_D == 0:
            lowers = np.array([phi_lo, np.log(ca_lo), np.log(ce_lo), b_lo])
            uppers = np.array([phi_up, np.log(ca_up), np.log(ce_up),  b_up])
        else:
            D_lo, D_up = self.lb[4], self.ub[4]
            lowers = np.array([phi_lo, np.log(ca_lo), np.log(ce_lo), b_lo, D_lo])
            uppers = np.array([phi_up, np.log(ca_up), np.log(ce_up),  b_up, D_up])

        Ddim = len(lowers)
        pop = np.zeros((population_size, Ddim))
        for i in range(Ddim):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], size=population_size)
        return pop, lowers, uppers

    # ----------------------
    # One-step forecast used by EnKF
    # ----------------------
    def model_step(self, par: np.ndarray, t_idx: int, context: Any | None = None):
        # Map transformed -> physical
        phi = par[0]
        ca  = float(np.exp(par[1]))
        ce  = float(np.exp(par[2]))
        b   = par[3]
        D   = (par[4] if self.switch_D == 1 else 2.0 * phi)

        # segment indices for this obs step
        i0, i1 = self.idx_obs_splited[t_idx - 1], self.idx_obs_splited[t_idx]
        P_seg    = self.P_s[i0:i1]
        Omega_seg= self.Omega_s[i0:i1]
        dt_seg   = self.dt_s[i0:i1]

        # initial condition for this segment
        y0 = float(self.Yini) if (context is None or ('y_old' not in context)) else float(context['y_old'])

        Ymd, _ = shoreFor_Yini(P_seg, Omega_seg, dt_seg, phi, D, ca, ce, b, y0)
        y_last = float(Ymd[-1])
        context = {'y_old': y_last}
        return y_last, context

    # ----------------------
    # Vectorized batch step (speeds EnKF a lot)
    # ----------------------
    def model_step_batch(self, pop: np.ndarray, t_idx: int, contexts: list[dict] | None):
        N = pop.shape[0]
        y_out   = np.empty((N,), dtype=float)
        new_ctx = [None] * N

        i0, i1 = self.idx_obs_splited[t_idx - 1], self.idx_obs_splited[t_idx]
        P_seg     = self.P_s[i0:i1]
        Omega_seg = self.Omega_s[i0:i1]
        dt_seg    = self.dt_s[i0:i1]

        for j in range(N):
            phi = pop[j, 0]
            ca  = float(np.exp(pop[j, 1]))
            ce  = float(np.exp(pop[j, 2]))
            b   = pop[j, 3]
            D   = (pop[j, 4] if self.switch_D == 1 else 2.0 * phi)

            y0 = float(self.Yini) if (contexts is None or contexts[j] is None
                                      or ('y_old' not in contexts[j])) else float(contexts[j]['y_old'])

            Ymd, _ = shoreFor_Yini(P_seg, Omega_seg, dt_seg, phi, D, ca, ce, b, y0)
            y_last = float(Ymd[-1])
            y_out[j] = y_last
            new_ctx[j] = {'y_old': y_last}

        return y_out, new_ctx

    # ----------------------
    # Full forward run with final parameters (for plotting/output)
    # ----------------------
    def run_model(self, par: np.ndarray) -> np.ndarray:
        # Here par is already in PHYSICAL space (after _set_parameter_names)
        phi, ca, ce, b = par[0], par[1], par[2], par[3]
        D = par[4] if self.switch_D == 1 else 2.0 * phi

        Ymd, _ = shoreFor_Yini(self.P, self.Omega, self.dt, phi, D, ca, ce, b, self.Yini)
        return Ymd

    # ----------------------
    # Naming and conversion to physical values for final reporting
    # ----------------------
    def _set_parameter_names(self):
        if self.switch_D == 0:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D']
            # self.par_values currently in transformed space; convert to physical
            phi  = float(self.par_values[0])
            ca   = float(np.exp(self.par_values[1]))
            ce   = float(np.exp(self.par_values[2]))
            b    = float(self.par_values[3])
            D    = 2.0 * phi
            self.par_values = np.array([phi, ca, ce, b, D], dtype=float)
        else:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D']
            phi  = float(self.par_values[0])
            ca   = float(np.exp(self.par_values[1]))
            ce   = float(np.exp(self.par_values[2]))
            b    = float(self.par_values[3])
            D    = float(self.par_values[4])
            self.par_values = np.array([phi, ca, ce, b, D], dtype=float)
