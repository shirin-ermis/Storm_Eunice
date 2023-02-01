'''
Functions to import data and metadata
'''

import xarray as xr
import os
import numpy as np
import pandas as pd

class Data():
    """
    Class to import data files
    """

    def __init__(self):
        self.image = None

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

        directory, experiments, inits, cfpf = Data.load_meta()
        experiment = 'pi'
        init = inits['pi'][0]
        cont = 'cf'

        lat = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc')).latitude.values
        lon = xr.open_dataset(os.path.join(directory[experiment],'EU025/sfc',cont,init+'.nc')).longitude.values
        return lat, lon

    def create_latlon_grid(lat, lon): 
        
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
        