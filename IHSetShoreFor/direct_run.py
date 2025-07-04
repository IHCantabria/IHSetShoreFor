import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .shoreFor import shoreFor_idx_Yini, shoreFor_Yini
import json
from IHSetUtils import wMOORE, BreakingPropagation
from scipy.stats import circmean

class ShoreFor_run(object):
    """
    cal_ShoreFor_2
    
    Configuration to calibrate and run the Davidson et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path):

        self.path = path
        self.name = 'ShoreFor (Davidson et al., 2013)'
        self.mode = 'standalone'
        self.type = 'CS'
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['run_ShoreFor'])
        self.cfg = cfg

        self.switch_Yini = cfg['switch_Yini']
        self.switch_D = cfg['switch_D']
        self.D50 = cfg['D50']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.breakType = cfg['break_type']



        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.tp = np.mean(data.tp.values, axis=1)
            self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
            self.depth = np.mean(data.waves_depth.values)
            self.bathy_angle = circmean(data.phi.values, high=360, low=0)
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.tp = data.tp.values[:, cfg['trs']]
            self.dir = data.dir.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.depth = data.waves_depth.values[cfg['trs']]
            self.bathy_angle = data.phi.values[cfg['trs']]
        
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

        if self.switch_Yini == 0 and self.switch_D == 0:
            def run_model(par):
                phi = par[0]
                cp = par[1]
                cm = par[2]
                D = par[3]
                Yini = [par[4]]

                Ymd, _ = shoreFor_Yini(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd
            
            self.run_model = run_model

        elif self.switch_Yini == 1 and self.switch_D == 0:
            def run_model(par):
                phi = par[0]
                cp = par[1]
                cm = par[2]
                D = par[3]

                Ymd, _ = shoreFor_idx_Yini(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.Obs,
                                    self.idx_obs)

                return Ymd
            
            self.run_model = run_model

        elif self.switch_Yini == 0 and self.switch_D == 1:
            def run_model(par):
                phi = par[0]
                cp = par[1]
                cm = par[2]
                D = 2 * phi
                Yini = [par[3]]

                Ymd, _ = shoreFor_Yini(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)

                return Ymd
            
            self.run_model = run_model

        elif self.switch_Yini == 1 and self.switch_D == 1:
            def run_model(par):
                phi = par[0]
                cp = par[1]
                cm = par[2]
                D = 2 * phi

                Ymd, _ = shoreFor_idx_Yini(self.P,
                                    self.Omega,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.Obs,
                                    self.idx_obs)
                return Ymd
            
            self.run_model = run_model
        
    def run(self, par):
        self.full_run = self.run_model(par)
        if self.switch_Yini == 1 and self.switch_D == 1:
            self.par_names = [r'phi', r'c_a', r'c_e']
            self.par_values = par
        elif self.switch_Yini == 0 and self.switch_D == 1:
            self.par_names = [r'phi', r'c_a', r'c_e', r'Y_i']
            self.par_values = par
        elif self.switch_Yini == 1 and self.switch_D == 0:
            self.par_names = [r'phi', r'c_a', r'c_e', r'D']
            self.par_values = par
        elif self.switch_Yini == 0 and self.switch_D == 0:
            self.par_names = [r'phi', r'c_a', r'c_e', r'D', r'Y_i']
            self.par_values = par

        # self.calculate_metrics()

    def calculate_metrics(self):
        self.metrics_names = fo.backtot()[0]
        self.indexes = fo.multi_obj_indexes(self.metrics_names)
        self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.P = self.P[ii]
        self.time = self.time[ii]
        self.Omega = self.Omega[ii]

        ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs = self.Obs[ii]
        self.time_obs = self.time_obs[ii]