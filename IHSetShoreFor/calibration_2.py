import numpy as np
from .shoreFor import shoreFor_idx_Yini, shoreFor_Yini
from IHSetUtils import wMOORE
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_ShoreFor_2(CoastlineModel):
    """
    cal_ShoreFor_2
    
    Configuration to calibrate and run the Davidson et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='ShoreFor (Davidson et al., 2013)',
            mode='calibration',
            model_type='CS',
            model_key='ShoreFor'
        )

        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        self.switch_D = self.cfg['switch_D']
        self.D50 = self.cfg['D50']

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]
                
        self.hb[self.hb < 0.1] = 0.1
        self.depthb[self.depthb < 0.2] = 0.2
        self.tp[self.tp < 2] = 2

        self.P = self.hb ** 2 * self.tp
        self.P_s = self.hb_s ** 2 * self.tp_s
        self.ws = wMOORE(self.D50)
        self.Omega = self.hb / (self.ws * self.tp)
        self.Omega_s = self.hb_s / (self.ws * self.tp_s)

    def init_par(self, population_size: int):
        if self.switch_Yini == 0 and self.switch_D == 0:
            lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), -1])
            uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1])
        elif self.switch_Yini == 1 and self.switch_D == 0:
            lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), -1, 0.75 * np.min(self.Obs)])
            uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1, 1.25 * np.max(self.Obs)])
        elif self.switch_Yini == 0 and self.switch_D == 1:
            lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), -1, self.lb[3]])
            uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1, self.ub[3]])
        elif self.switch_Yini == 1 and self.switch_D == 1:
            lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), -1, self.lb[3], 0.75 * np.min(self.Obs)])
            uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1, self.ub[3], 1.25 * np.max(self.Obs)])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers
    
    def model_sim(self, par: np.ndarray) -> np.ndarray:
        phi = par[0]
        cp = np.exp(par[1])
        cm = np.exp(par[2])
        b = par[3]
        if self.switch_Yini == 0 and self.switch_D == 0:
            D = 2 * phi
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_D == 0:
            D = 2 * phi
            Yini = par[4]
        elif self.switch_Yini == 0 and self.switch_D == 1:
            D = par[4]
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_D == 1:
            D = par[4]
            Yini = par[5]
        
        Ymd, _ = shoreFor_Yini(self.P_s,
                                self.Omega_s,
                                self.dt_s,
                                phi,
                                D,
                                cp,
                                cm,
                                b,
                                Yini)
        return Ymd[self.idx_obs_splited]

    def run_model(self, par: np.ndarray) -> np.ndarray:
        phi = par[0]
        cp = par[1]
        cm = par[2]
        b = par[3]
        D = par[4]
        if self.switch_Yini == 0:
            Yini = self.Yini
        elif self.switch_Yini == 1:
            Yini = par[5]
        Ymd, _ = shoreFor_Yini(self.P,
                                self.Omega,
                                self.dt,
                                phi,
                                D,
                                cp,
                                cm,
                                b,
                                Yini)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Yini == 0 and self.switch_D == 0:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D']
            self.par_values = np.hstack((self.par_values, 2*self.par_values[0]))
        elif self.switch_Yini == 1 and self.switch_D == 0:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D', r'Y_i']
            aux = np.hstack((self.par_values[:-1], 2*self.par_values[0]))
            self.par_values = np.hstack((aux, self.par_values[-1]))
        elif self.switch_Yini == 0 and self.switch_D == 1:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D']
        elif self.switch_Yini == 1 and self.switch_D == 1:
            self.par_names = [r'phi', r'c_a', r'c_e', r'b', r'D', r'Y_i']

        for idx in [1, 2]:
            self.par_values[idx] = np.exp(self.par_values[idx])
