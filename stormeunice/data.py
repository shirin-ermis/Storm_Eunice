'''
Functions to import data and metadata
'''

import xarray as xr
import os
import numpy as np
import pandas as pd

import scipy as sp
import netCDF4 as nC4
import scipy.signal
import sys
import glob
import datetime
import time
import multiprocessing
import copy
import shutil
import gzip
import warnings
import glob


class Data():
    """
    Class to import data files and pre-process them.
    """

    def __init__(self):
        self.status = None

    def load_meta():
        """
        Function loads the data in directory dir and file fielname

        Input
        -----
        dir: str, path to directory
        filename: str, filename of data

        Output
        ------
        directory, experiments, inits, cfpf 
        """

        # Load data from MED-R Preindustrial and increased, as before
        directory = {'pi' : '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/pi',
                    'incr' : '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr', 
                    'curr' : '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/'}

        experiments = ['incr', 'pi', 'curr']
        cfpf = ['cf', 'pf']  # control and perturbed are treated equally
        inits = {'pi' : ['b2nq_2022-02-10', 'b2nn_2022-02-14', 'b2ns_2022-02-16'], 
                'incr' : ['b2nr_2022-02-10', 'b2no_2022-02-14', 'b2nt_2022-02-16'], 
                'curr' : ['1_2022-02-10', '1_2022-02-14', '1_2022-02-16']}  # members for incr and pi runs

        return directory, experiments, inits, cfpf
    
    def get_latlon():
        """
        Function loads latitude and longitude grid for forecasts when called.

        Input:
        ------

        Output:
        ------
        lat, lon: arrays, latitude and longitude arrays
        """

        directory, experiments, inits, cfpf = Data.load_meta()
        experiment = 'pi'
        init = inits['pi'][0]
        cont = 'cf'

        lat = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc')).latitude.values
        lon = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc')).longitude.values
        return lat, lon

    def create_latlon_grid(lat, lon): 
        """
        Creates a meshgrid for latitude and longitude for the south of England

        Input:
        ------

        Output:
        -------
        llat: 2d array, grid of latitude values
        llon: 2d array, grid of longitude values
        """
        
        # Defining box to analyse winds, south england and wales
        lat1 = 52.2
        lat2 = 50.3
        lon1 = -6
        lon2 = 1.3

        # create meshgrid
        south_england_dict = {'lat': (lat < lat1) & (lat > lat2), 'lon': (lon < lon2) & (lon > lon1)}
        llat, llon = np.meshgrid(lon[south_england_dict['lon']], lat[south_england_dict['lat']])

        return llat, llon

    def get_friday_data(): 
        """
        Function to load data for Friday, 18th February 2022 for all ensembles and experiments

        Input:
        -------

        Output:
        -------
        south_df: tidy pandas data frame with data from South of UK on Friday
        """

        lat, lon = Data.get_latlon()
        directory, experiments, inits, cfpf = Data.load_meta()
        # Defining box to analyse winds, south england and wales
        lat1 = 52.2
        lat2 = 50.3
        lon1 = -6
        lon2 = 1.3
        llat, llon = Data.create_latlon_grid(lat, lon)

        filename = './Eunice_Friday_lat-'+str(lat1)+'-'+str(lat2)+'_lon-'+str(lon1)+'-'+str(lon2)+'.csv'

        if os.path.isfile(filename): 
            south_df = pd.read_csv(filename)
        else: 
            # empty data frame to be filled later
            south_df = pd.DataFrame({'lat': [], 
                                    'lon' : [],  
                                    'experiment' :[],
                                    'cfpf' : [], 
                                    'member' : [], 
                                    'init' : [],
                                    'time' : [],
                                    'fg10' : []})

            # Fill data frame
            members = 50
            for experiment in experiments:
                for init in inits[experiment]:
                    for cont in cfpf:
                        
                        # import full data set in file
                        data = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc'))
                        south_england = (data.latitude < lat1) & (data.latitude > lat2) & (data.longitude < lon2) & (data.longitude > lon1) 
                        friday = (data.time >= pd.Timestamp(2022,2,18, 0)) & (data.time <= pd.Timestamp(2022,2,18, 18))
                        data_filtered = data.where(south_england & friday, drop = True)
                        
                        # store data in data frame with meta data
                        if cont == 'cf':  # distinguish between cfpf because members in pf
                            length = len(data_filtered.fg10.values.flatten())
                            number_timesteps = len(data_filtered.time.values)
                            n_lat = llat.shape[0]
                            n_lon = llon.shape[1]
                            adding = pd.DataFrame({'lat': np.tile(llat.flatten(), number_timesteps), 
                                                    'lon' : np.tile(llon.flatten(), number_timesteps), 
                                                    'experiment' : np.tile(experiment, length),
                                                    'cfpf' : np.tile(cont, length), 
                                                    'member' : np.tile(-1, length), 
                                                    'init' : np.tile(init, length), 
                                                    'time': np.repeat(data_filtered.time.values.flatten(), n_lat*n_lon), 
                                                    'fg10' : data_filtered.fg10.values.flatten()})
                            south_df = pd.merge(south_df, adding,
                                                how = 'outer')
                        
                        elif cont == 'pf': 
                            for member in range(members):
                                n_lat = llat.shape[0]
                                n_lon = llon.shape[1]
                                adding = pd.DataFrame({'lat': np.tile(llat.flatten(), number_timesteps), 
                                                        'lon' : np.tile(llon.flatten(), number_timesteps), 
                                                        'experiment' : np.tile(experiment, length),
                                                        'cfpf' : np.tile(cont, length), 
                                                        'member' : np.tile(member, length), 
                                                        'init' : np.tile(init, length), 
                                                        'time': np.repeat(data_filtered.time.values.flatten(), n_lat*n_lon), 
                                                        'fg10' : data_filtered.fg10.values[:,member,:,:].flatten()})
                                south_df = pd.merge(south_df, adding,
                                                    how = 'outer')
            south_df.to_csv(filename)

        return south_df

    def  get_friday_data_xr():  # TODO
        """
        Function to load data for Friday, 18th February 2022 for all ensembles and experiments
        Output here is an xarray

        Input:
        -------

        Output:
        -------
        south_xr: xarray with data from South of UK on Friday
        """

        lat, lon = Data.get_latlon()
        directory, experiments, inits, cfpf = Data.load_meta()
        # Defining box to analyse winds, south england and wales
        lat1 = 52.2
        lat2 = 50.3
        lon1 = -6
        lon2 = 1.3
        llat, llon = Data.create_latlon_grid(lat, lon)

        filename = './Eunice_Friday_lat-'+str(lat1)+'-'+str(lat2)+'_lon-'+str(lon1)+'-'+str(lon2)+'.csv'

        if os.path.isfile(filename): 
            south_xr = xr.load_dataset(filename)
        else:
            # Fill data frame
            members = 50
            for experiment in experiments:
                for init in inits[experiment]:
                    for cont in cfpf:
                        
                        # import full data set in file
                        data = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc'))
                        
        return south_xr

    def accum2rate(ds):
        """
        Function to convert accumulated variables to conventional ones. 
        Definition to convert accumulated variables to instantaneous:. 
        Written by Nick Leach.

        Input:
        ------

        Output:
        -------
        """
        ## accumulated variables & scaling factors
        accumulated_vars = {'tp':60 * 60 * 24 * 1e3,'ttr':1,'tsr':1,'str':1,'ssr':1,'e':1}
        accumulated_var_newunits = {'tp':'mm day$^{-1}$','ttr':'W m$^{-2}$','tsr':'W m$^{-2}$','str':'W m$^{-2}$','ssr':'W m$^{-2}$','e':'m s$^{-1}$'}

        ds = ds.copy()
        oindex = ds.time
        inidate = pd.to_datetime(oindex[0].values)
        
        ds = ds.diff('time') / (ds.time.diff('time').astype(float) / 1e9 )
        ds = ds.reindex(time=oindex)
        return ds[1:]

    def preproc_ds(ds):
        """
        Main pre-processing function
        Writtten by Nick Leach.

        Input:
        ------

        Output:
        -------
        """
    
        ## accumulated variables & scaling factors
        accumulated_vars = {'tp':60 * 60 * 24 * 1e3,'ttr':1,'tsr':1,'str':1,'ssr':1,'e':1}
        accumulated_var_newunits = {'tp':'mm day$^{-1}$','ttr':'W m$^{-2}$','tsr':'W m$^{-2}$','str':'W m$^{-2}$','ssr':'W m$^{-2}$','e':'m s$^{-1}$'}

    
        ds = ds.copy().squeeze()
        fname = ds.encoding['source'].split('/')[-1].split('.')[0]
        expver = fname.split('_')[0]
        ds = ds.expand_dims({'experiment':[expver]}).copy()

        # set up aux data
        inidate = pd.to_datetime(ds.time[0].values)
        
        # expand dimensions to include extra info
        if not 'hDate' in ds:
            ds = ds.expand_dims({'inidate':[inidate]}).copy()
            
        if not 'number' in ds:
            ds = ds.expand_dims({'number':[0]}).copy()
            
        # put time dimension at front
        ds = ds.transpose('time',...)
        ds = ds.copy(deep=True)
        
        # convert accumulated variables into instantaneous
        for var,sf in accumulated_vars.items():
            if var in ds.keys():
                ds[var].loc[dict(time =ds.time[1:])] = Data.accum2rate(ds[var]) * sf
                # set first value to equal zero [since it should be zero... but isn't always]
                ds[var].loc[dict(time=ds.time[0])] = 0
                ds[var].attrs['units'] = accumulated_var_newunits[var]
                
        return ds
    
    def preproc_mclim(ds):
        """
        A couple more steps for pre-processing m-climate
        Written by Nick Leach.

        Input:
        ------
        ds: xarray

        Output:
        -------
        ds: xarray
        """
    
        ds = ds.copy().squeeze()
        ds = Data.preproc_ds(ds)
        # create index of hours from initialisation
        ds_hours = ((ds.time-ds.time.isel(time=0))/1e9/3600).astype(int)
        # change time coord to hours coord + rename
        ds = ds.assign_coords(time=ds_hours).rename(dict(time='hour'))
        
        return ds


    def get_eps_data():  # TODO
        """
        Function to load comlete data of operational forecast since xr has a bug that prevents using 
        this as a simpler solution

        Input:
        ------
        none

        Output:
        -------
        eps: xarray, data and metadata of operational forecasts
        """

        directory = '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/sfc/*/*.nc'
        ind = 0
        for files in glob.glob(directory):
            if ind == 0:
                eps = Data.preproc_ds(xr.open_dataset(files))
                ind += 1
            else:
                data = xr.open_dataset(files)
                preproc_data = Data.preproc_ds(data)
                eps = xr.combine_by_coords(eps,preproc_data)
        
        return eps
            