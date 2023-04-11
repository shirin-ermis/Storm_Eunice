'''
Functions to import data and metadata
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
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
        directory = {'pi': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/pi',
                     'incr': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr',
                     'curr': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/'}

        experiments = ['incr', 'pi', 'curr']
        cfpf = ['cf', 'pf']  # control and perturbed are treated equally
        inits = {'pi': ['b2nq_2022-02-10', 'b2nn_2022-02-14', 'b2ns_2022-02-16'],
                 'incr': ['b2nr_2022-02-10', 'b2no_2022-02-14', 'b2nt_2022-02-16'],
                 'curr': ['1_2022-02-10', '1_2022-02-14', '1_2022-02-16']}

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

        lat = xr.open_dataset(os.path.join(directory[experiment], 'EU025/sfc', cont, init + '.nc')).latitude.values
        lon = xr.open_dataset(os.path.join(directory[experiment], 'EU025/sfc', cont, init + '.nc')).longitude.values
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

        filename = './Eunice_Friday_lat-' + str(lat1) + '-' + str(lat2) + '_lon-' + str(lon1) + '-' + str(lon2) + '.csv'

        if os.path.isfile(filename):
            south_df = pd.read_csv(filename)
        else:
            # empty data frame to be filled later
            south_df = pd.DataFrame({'lat': [],
                                     'lon': [],
                                     'experiment': [],
                                     'cfpf': [],
                                     'member': [],
                                     'init': [],
                                     'time': [],
                                     'fg10': []})

            # Fill data frame
            members = 50
            for experiment in experiments:
                for init in inits[experiment]:
                    for cont in cfpf:
                        
                        # import full data set in file
                        data = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc', cont, init + '.nc'))
                        south_england = (data.latitude < lat1) & (data.latitude > lat2) & (data.longitude < lon2) & (data.longitude > lon1)
                        friday = (data.time >= pd.Timestamp(2022, 2, 18, 0)) & (data.time <= pd.Timestamp(2022, 2, 18, 18))
                        data_filtered = data.where(south_england & friday, drop=True)
                        
                        # store data in data frame with meta data
                        # distinguish between cfpf because members in pf
                        if cont == 'cf':
                            length = len(data_filtered.fg10.values.flatten())
                            number_timesteps = len(data_filtered.time.values)
                            n_lat = llat.shape[0]
                            n_lon = llon.shape[1]
                            adding = pd.DataFrame({'lat': np.tile(llat.flatten(), number_timesteps),
                                                   'lon': np.tile(llon.flatten(), number_timesteps),
                                                   'experiment': np.tile(experiment, length),
                                                   'cfpf': np.tile(cont, length),
                                                   'member': np.tile(-1, length),
                                                   'init': np.tile(init, length),
                                                   'time': np.repeat(data_filtered.time.values.flatten(), n_lat * n_lon),
                                                   'fg10': data_filtered.fg10.values.flatten()})
                            south_df = pd.merge(south_df, adding,
                                                how='outer')
                        elif cont == 'pf':
                            for member in range(members):
                                n_lat = llat.shape[0]
                                n_lon = llon.shape[1]
                                adding = pd.DataFrame({'lat': np.tile(llat.flatten(), number_timesteps),
                                                        'lon': np.tile(llon.flatten(), number_timesteps),
                                                        'experiment': np.tile(experiment, length),
                                                        'cfpf': np.tile(cont, length),
                                                        'member': np.tile(member, length),
                                                        'init': np.tile(init, length),
                                                        'time': np.repeat(data_filtered.time.values.flatten(), n_lat * n_lon),
                                                        'fg10': data_filtered.fg10.values[:, member, :, :].flatten()})
                                south_df = pd.merge(south_df, adding,
                                                    how='outer')
            south_df.to_csv(filename)

        return south_df

    def accum2rate(ds):
        """
        Function to convert accumulated variables to conventional ones.
        Definition to convert accumulated variables to instantaneous.
        Written by Nick Leach.

        Input:
        ------

        Output:
        -------
        """
        # accumulated variables & scaling factors
        accumulated_vars = {'tp': 60 * 60 * 24 * 1e3,
                            'ttr': 1,
                            'tsr': 1,
                            'str': 1,
                            'ssr': 1,
                            'e': 1}
        accumulated_var_newunits = {'tp': 'mm day$^{-1}$',
                                    'ttr': 'W m$^{-2}$',
                                    'tsr': 'W m$^{-2}$',
                                    'str': 'W m$^{-2}$',
                                    'ssr': 'W m$^{-2}$',
                                    'e':'m s$^{-1}$'}

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

        # accumulated variables & scaling factors
        accumulated_vars = {'tp': 60 * 60 * 24 * 1e3,
                            'ttr': 1,
                            'tsr': 1,
                            'str': 1,
                            'ssr': 1,
                            'e': 1}
        accumulated_var_newunits = {'tp': 'mm day$^{-1}$',
                                    'ttr': 'W m$^{-2}$',
                                    'tsr': 'W m$^{-2}$',
                                    'str': 'W m$^{-2}$',
                                    'ssr': 'W m$^{-2}$',
                                    'e': 'm s$^{-1}$'}
        ds = ds.copy().squeeze()
        # set up aux data
        inidate = pd.to_datetime(ds.time[0].values)
        # expand dimensions to include extra info
        if not 'hDate' in ds:
            ds = ds.expand_dims({'inidate': [inidate]}).copy()

        if not 'number' in ds:
            ds = ds.expand_dims({'number': [0]}).copy()

        # put time dimension at front
        ds = ds.transpose('time', ...)
        ds = ds.copy(deep=True)

        # convert accumulated variables into instantaneous
        for var, sf in accumulated_vars.items():
            if var in ds.keys():
                ds[var].loc[dict(time=ds.time[1:])] = Data.accum2rate(ds[var]) * sf
                # set first value to equal zero,
                # should be zero but isn't always
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
        ds_hours = ((ds.time - ds.time.isel(time=0)) / 1e9 / 3600).astype(int)
        # change time coord to hours coord + rename
        ds = ds.assign_coords(time=ds_hours).rename(dict(time='hour'))
        return ds

    def get_eps_data(experiments, inidate='2022-02-16'):
        """
        Function to load comlete data of simulations on surface level since
        xr has a bug that prevents using
        this as a simpler solution

        Input:
        ------
        experiments: list of strings, list of experiments to import,
                    e.g. ['pi', 'curr', 'incr']

        Output:
        -------
        eps: list of xarrays, data and metadata of operational forecasts,
            each list entry is one experiment
        """

        directory = {'pi': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/pi/EU025/sfc/',
                     'curr': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/sfc/',
                     'incr': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr/EU025/sfc/'}

        eps = {}
        for experiment in experiments:
            exp_eps = []
            for c, cont in enumerate(['cf', 'pf']):
                for files in glob.glob(directory[experiment] + cont + '/*' + inidate + '*.nc'):
                    print(files)
                    data = xr.open_dataset(files)
                    exp_eps.append(Data.preproc_ds(data.get(['fg10', 'msl', 'u10', 'v10', 'u100', 'v100'])))

            eps[experiment] = xr.concat(exp_eps, dim='number').squeeze()

        return eps

    def get_eps_pl_data(experiments, inidate='2022-02-16', level=500):
        """
        Function to load comlete data of simulations on pressure levels
        since xr has a bug that prevents using
        this as a simpler solution

        Input:
        ------
        experiments: list of strings, list of experiments to import,
            e.g. ['pi', 'curr', 'incr']
        level: int, pressure level in hPa that data is on

        Output:
        -------
        eps: dictionary of xarrays, data and metadata of operational forecasts,
            each list entry is one experiment
        """

        eps = {}

        for experiment in experiments:

            directory = {'pi': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/pi/EU025/pl/',
                         'curr': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/pl/',
                         'incr': '/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr/EU025/pl/'}

            exp_eps = []
            for c, cont in enumerate(['cf', 'pf']):
                for files in glob.glob(directory[experiment] + cont + '/*' + inidate + '*.nc'):
                    print(files)
                    data = xr.open_dataset(files)
                    data = Data.preproc_ds(data.sel(level=level).get(['z', 'vo']))  # preprocessing just two variables for speed
                    exp_eps.append(Data.preproc_ds(xr.open_dataset(files).get(['z', 'vo'])))

                eps[experiment] = xr.concat(exp_eps, dim='number').squeeze()

        return eps

    def get_era_98thperc_winds(height=100):
        """
        Function to load the 98th percentile of wind speeds in
        ERA5 for the period 2010-2019.

        Input:
        ------
        height: int, level in m (10 or 100) at which wind speeds and
            percentiles are calculated

        Output:
        -------
        era5_windspeeds_98perc: xarray dataset of 98th percentile of wind
            gusts for all grid points over Europe
        """

        filename = 'era5_2010-2019_windspeeds' + str(height) + '_98thperc.nc'

        if os.path.isfile(filename):
            era5_windspeeds_98perc = xr.open_dataset(filename)
        else:
            era5_2000_2022 = xr.open_mfdataset('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/ERA5/EU025/sfc/201*.nc',
                                               chunks=dict(time=-1)).get(['u10', 'v10', 'v100', 'u100'])
            if height == 100:
                era5_windspeeds = era5_2000_2022.assign(ws100=(era5_2000_2022.u100**2 + era5_2000_2022.v100**2)**(1 / 2))
                era5_windspeeds_98perc = era5_windspeeds.chunk(dict(time=-1)).ws100.quantile(0.98, dim = ['time'])
            elif height == 10:
                era5_windspeeds = era5_2000_2022.assign(ws10=(era5_2000_2022.u10**2 + era5_2000_2022.v10**2)**(1 / 2))
                era5_windspeeds_98perc = era5_windspeeds.chunk(dict(time=-1)).ws10.quantile(0.98, dim=['time'])
            else:
                Exception(ValueError("height must be 10 or 100."))
            era5_windspeeds_98perc.to_netcdf(filename)

        return era5_windspeeds_98perc

    def get_era_98thperc_gusts():
        """
        Function to load the 98th percentile of wind speeds in ERA5 for the
        period 2010-2019.

        Input:
        ------

        Output:
        -------
        era5_windgusts_98perc: xarray dataset of 98th percentile of wind gusts
            for all grid points over Europe
        """

        filename = 'era5_2010-2019_windsgusts_98thperc.nc'

        if os.path.isfile(filename):
            era5_windgusts_98perc = xr.open_dataset(filename)
        else:
            era5_windgusts = xr.open_mfdataset('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/ERA5/EU025/sfc/201*.nc', chunks=dict(time=-1)).get(['fg10'])
            era5_windgusts_98perc = era5_windgusts.chunk(dict(time=-1)).fg10.quantile(0.98, dim=['time'])
            era5_windgusts_98perc.to_netcdf(filename)

        return era5_windgusts_98perc

    def get_eps_windpseeds(arr):

        arrWindspeeds = arr.assign(ws10=(arr.v10**2 + arr.u10**2)**(1 / 2),
                                   ws100=(arr.v100**2 + arr.u100**2)**(1 / 2))
        return arrWindspeeds
