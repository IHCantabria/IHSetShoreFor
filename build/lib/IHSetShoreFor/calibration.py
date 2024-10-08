import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from .shoreFor import shoreFor
from IHSetCalibration import objective_functions
from IHSetUtils import wMOORE, BreakingPropagation

class cal_ShoreFor(object):
    """
    cal_ShoreFor
    
    Configuration to calibrate and run the Davidson et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = int(cfg['dt'].values)
        self.switch_Yini = cfg['switch_Yini'].values
        self.switch_D = cfg['switch_D'].values
        self.switch_brk = cfg['switch_brk'].values
        # self.switch_r = cfg['switch_r'].values
        self.D50 = cfg['D50'].values
        
        
        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values

        if self.switch_brk == 0:
            self.Hb = self.Hs
        else:
            self.depth = np.zeros_like(self.Hs) + cfg['depth'].values
            self.bathy_angle = np.zeros_like(self.Hs) + cfg['bathy_angle'].values
            self.breaking_type = cfg['breaking_type'].values
            self.Hb, _, _ = BreakingPropagation(self.Hs, self.Tp, self.Dir, self.depth, self.bathy_angle, self.breaking_type)

        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        
        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.P = self.Hb ** 2 * self.Tp
        self.ws = wMOORE(self.D50)
        self.Omega = self.Hs / (self.ws * self.Tp)
        
        self.split_data()

        cfg.close()
        wav.close()
        ens.close()

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        # if self.switch_Yini == 0:
        #     self.Yini = self.Obs_splited[0]

        if self.switch_Yini == 0 and self.switch_D == 0:
            def model_simulation(par):
                phi = par['phi']
                cp = par['cp']
                cm = par['cm']
                D = 2 * phi

                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs_splited)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('phi', 3, 365),
                Uniform('cp', 1e-6, 1e-2),
                Uniform('cm', 1e-6, 1e-2)
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1 and self.switch_D == 0:
            def model_simulation(par):
                phi = par['phi']
                cp = par['cp']
                cm = par['cm']
                D = 2 * phi
                Yini = par['Yini']
                
                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('phi', 3, 365),
                Uniform('cp', 1e-6, 1e-2),
                Uniform('cm', 1e-6, 1e-2),
                Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 0 and self.switch_D == 1:
            def model_simulation(par):
                phi = par['phi']
                cp = par['cp']
                cm = par['cm']
                D = par['D']
                N = par['N']
                
                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    self.observations,
                                    self.idx_obs_splited)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('phi', 3, 365),
                Uniform('cp', 1e-6, 1e-2),
                Uniform('cm', 1e-6, 1e-2),
                Uniform('D', 6, 730)
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1 and self.switch_D == 1:
            def model_simulation(par):
                phi = par['phi']
                cp = par['cp']
                cm = par['cm']
                D = par['D']
                Yini = par['Yini']


                                
                Ymd, _ = shoreFor(self.P_splited,
                                    self.Omega_splited,
                                    self.dt,
                                    phi,
                                    D,
                                    cp,
                                    cm,
                                    Yini)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('phi', 3, 365),
                Uniform('cp', 1e-6, 1e-2),
                Uniform('cm', 1e-6, 1e-2),
                Uniform('D', 6, 730),
                Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
            ]
            self.model_sim = model_simulation
        
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


        # if self.switch_Yini == 0 and self.switch_D == 0 and self.switch_r == 0:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = 2 * phi

        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             self.Yini, 
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2)
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 1 and self.switch_D == 0 and self.switch_r == 0:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = 2 * phi
        #         Yini = par['Yini']
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             Yini, 
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 0 and self.switch_D == 1 and self.switch_r == 0:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = par['D']
        #         N = par['N']
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             self.Yini, 
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('D', 6, 730)
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 1 and self.switch_D == 1 and self.switch_r == 0:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = par['D']
        #         Yini = par['Yini']
                                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             Yini, 
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('D', 6, 730),
        #         Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
        #     ]
        #     self.model_sim = model_simulation
        
        # elif self.switch_Yini == 0 and self.switch_D == 0 and self.switch_r == 1:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = 2 * phi
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             self.Yini,
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 1 and self.switch_D == 0 and self.switch_r == 1:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = 2 * phi
        #         Yini = par['Yini']
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             Yini,
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 0 and self.switch_D == 1 and self.switch_r == 1:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = par['D']
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             self.Yini,
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('D', 6, 730)
        #     ]
        #     self.model_sim = model_simulation

        # elif self.switch_Yini == 1 and self.switch_D == 1 and self.switch_r == 1:
        #     def model_simulation(par):
        #         phi = par['phi']
        #         cp = par['cp']
        #         cm = par['cm']
        #         D = par['D']
        #         Yini = par['Yini']
                
        #         Ymd, _ = shoreFor(self.P_splited,
        #                             self.Omega_splited,
        #                             self.dt,
        #                             phi,
        #                             D,
        #                             Yini,
        #                             cp,
        #                             cm)
        #         return Ymd[self.idx_obs_splited]
            
        #     self.params = [
        #         Uniform('phi', 3, 365),
        #         Uniform('cp', 1e-6, 1e-2),
        #         Uniform('cm', 1e-6, 1e-2),
        #         Uniform('D', 6, 730),
        #         Uniform('Yini', 0.5*np.min(self.Obs), 1.5*np.max(self.Obs))
        #     ]
        #     self.model_sim = model_simulation
