import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .shoreFor import shoreFor
import json
from IHSetUtils import wMOORE, BreakingPropagation
from scipy.stats import circmean

class cal_ShoreFor_2(object):
    """
    cal_ShoreFor_2
    
    Configuration to calibrate and run the Davidson et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['ShoreFor'])

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.switch_D = cfg['switch_D']
        self.lb = cfg['lb']
        self.ub = cfg['ub']
        self.D50 = cfg['D50']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.bathy_angle = cfg['bathy_angle']
            self.breakType = cfg['break_type']
            self.depth = cfg['depth']
        
        self.calibr_cfg = fo.config_cal(cfg)            

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.tp = np.mean(data.tp.values, axis=1)
            self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.tp = data.tp.values[:, cfg['trs']]
            self.dir = data.dir.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        data.close()
        
        if self.switch_brk == 0:
            self.depthb = self.hs / 0.78
            self.hb = self.hs
            self.dirb = self.dir
        elif self.switch_brk == 1:
            self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
        self.hb[self.hb < 0.1] = 0.1
        self.depthb[self.depthb < 0.2] = 0.2
        self.tp[self.tp < 5] = 5

        self.P = self.hb ** 2 * self.tp
        self.ws = wMOORE(self.D50)
        self.Omega = self.hb / (self.ws * self.tp)

        self.split_data()

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))


        if self.switch_Yini == 0 and self.switch_D == 0:
            def model_simulation(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = 2 * phi

                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt_splited,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs_splited)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]

                Ymd, _ = shoreFor(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2])])
                log_upper_bounds = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2])])
                population = np.zeros((population_size, 3))
                for i in range(3):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1 and self.switch_D == 0:
            def model_simulation(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = 2 * phi
                Yini = [par[3]]

                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt_splited,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]
                Yini = [par[4]]

                Ymd, _ = shoreFor(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), 0.75*np.min(self.Obs)])
                log_upper_bounds = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1.25*np.max(self.Obs)])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 0 and self.switch_D == 1:
            def model_simulation(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]

                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt_splited,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs_splited)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]

                Ymd, _ = shoreFor(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), self.lb[3]])
                log_upper_bounds = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), self.ub[3]])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par


        elif self.switch_Yini == 1 and self.switch_D == 1:
            def model_simulation(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]
                Yini = [par[4]]

                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt_splited,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                phi = par[0]
                cp = np.exp(par[1])
                cm = np.exp(par[2])
                D = par[3]
                Yini = [par[4]]

                Ymd, _ = shoreFor(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), self.lb[3], 0.75*np.min(self.Obs)])
                log_upper_bounds = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), self.ub[3], 1.25*np.max(self.Obs)])
                population = np.zeros((population_size, 5))
                for i in range(5):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par
        
    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where(self.time>=self.start_date)[0][0]
        self.P = self.P[ii:]
        self.time = self.time[ii:]
        self.Omega = self.Omega[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.P_splited = self.P[idx]
        self.Omega_splited = self.Omega[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]

        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]
        self.observations = self.Obs_splited

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)

        if self.switch_D == 0 and self.switch_Yini == 0:
            self.solution = np.array([self.solution[0], self.solution[1], self.solution[2]])
            self.solution = np.hstack((self.solution, 2*self.solution[0]))
        
        elif self.switch_D == 0 and self.switch_Yini == 1:
            self.solution = np.array([self.solution[0], self.solution[1], self.solution[2]])
            self.solution = np.hstack((self.solution, 2*self.solution[0]))
            self.solution = np.hstack((self.solution, self.solution[3]))