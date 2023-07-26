'''
Functions to preprocess data for lagrangian composites
'''

import pandas as pd
import numpy as np
import xarray as xr


class Lagrange():
    """
    Class to import data files and pre-process them as Lagrangian composites.
    """

    def lagrangian_frame(ds):
        '''
        Function to calculate Lagrangian frame from tracks
        Written by Nick Leach.

        Input:
        ------
        ds: xarray dataset

        Output:
        -------
        '''
        ds = ds.squeeze()
        ds = ds.assign_coords(latitude=ds.latitude - ds.centroid_lat,
                              longitude=ds.longitude - ds.centroid_lon)
        ds = ds.rename(latitude='storm_lat', longitude='storm_lon')
        ds = ds.sel(storm_lon=slice(-10, 10), storm_lat=slice(10, -10))
        return ds

    def import_medr_tracks_TE(fpath):
        '''
        Funtion toimport medium range tracks from
        Tempest Extremes algorithm
        Written by Nick Leach.

        Input:
        ------
        fpath: string, file path

        Output:
        -------
        '''

        df = pd.read_csv(fpath, skipinitialspace=True)

        expdict = {'1': 'ENS',
                   'b2nn': 'pi',
                   'b2nq': 'pi',
                   'b2ns': 'pi',
                   'b2no': 'incr',
                   'b2nr': 'incr',
                   'b2nt': 'incr'}

        fname = fpath.split('/')[-1]
        _, expid, inidate, mem = fname.split('_')

        df['expid'] = expid
        df['experiment'] = expdict[expid]
        df['inidate'] = pd.to_datetime(inidate)
        df['number'] = int(mem)
        return df

    def eunice_dist(df, eunice_track=None):
        '''
        Funtion to calculate the distance o the track.
        Written by Nick Leach.

        Input:
        ------
        df: pandas dataframe

        Output:
        -------
        '''

        eunice_lons = eunice_track.lon.values
        eunice_lats = eunice_track.lat.values

        track_lons = df.lon.values
        track_lats = df.lat.values

        minsize = min(eunice_lons.size, track_lons.size)

        return np.sqrt((track_lons[:minsize]
                        - eunice_lons[:minsize])**2
                       + (track_lats[:minsize]
                          - eunice_lats[:minsize])**2).sum()

    def preproc_to_stormframe(ds, ifs_eunice_list=None, sfc=True):
        '''
        Funtion for pre-processing to Lagrangian fields for tracked storms.
        Written by Nick Leach and Shirin Ermis.

        Input:
        ------
        ds: xarray dataset
        ifs_eunice_list: pandas dataframe of track information
        sfc: bool, whether surface data or pressure level data is needed
        level:

        Output:
        -------
        LG_fields: xarray dataset with Lagrangian fileds for Eunice like storms
        '''

        ds = ds.copy()

        if 'number' not in ds.coords:
            ds = ds.expand_dims({'number': [0]})

        fpath = ds.encoding['source']
        if sfc:
            exp = fpath.split('/')[-5]
        else:
            exp = fpath.split('/')[-6]
        inidate = fpath.split('/')[-1].split('_')[-1].split('.')[0]
        ds_tracks = ifs_eunice_list.query('experiment=="{}" & inidate=="{}"'.format(exp, inidate))
        LG_fields = []

        for num in set(ds.number.values).intersection(ds_tracks.number.unique()):

            mem_track = ds_tracks.loc[ds_tracks.number == num]
            mem_fields = ds.sel(number=num)
            time_intersection = sorted(list(set(mem_fields.time.values).intersection(mem_track.date.values)))

            resample_freq = 3  # resampling frequency in hours
            if inidate == '2022-02-10':
                resample_freq = 6

            # get start / end times for properly calculating the maximum
            # fields (taking into account the different preproc times in IFS)
            time_start = time_intersection[0] - pd.Timedelta('{}h 59m'.format(resample_freq - 1))
            time_end = time_intersection[-1]

            # get the instantaneous fields + wind speeds
            if sfc:
                mem_fields_out = mem_fields.get(['sst',
                                                 'u10',
                                                 'v10',
                                                 'msl',
                                                 'u100',
                                                 'v100',
                                                 'fg10',
                                                 'tcwv']).sel(time=time_intersection)
                mem_fields_out['ws10'] = np.sqrt(mem_fields_out.u10**2 + mem_fields_out.v10**2)
                mem_fields_out['ws100'] = np.sqrt(mem_fields_out.u100**2 + mem_fields_out.v100**2)

                # get the maximum fields, taking into account the different preproc times
                mxtpr_field_out = mem_fields.mxtpr.sel(time=slice(time_start, time_end)).resample(time='{}h'.format(resample_freq), label='right', closed='right', base=0).max()
                mem_fields_out['mxtpr'] = mxtpr_field_out
            else:
                mem_fields_out = mem_fields.get(['z',
                                                 'q',
                                                 'r',
                                                 'w',
                                                 't',
                                                 'd',
                                                 'u',
                                                 'v',
                                                 'r',
                                                 'vo']).sel(time=time_intersection)
                mem_fields_out['ws'] = np.sqrt(mem_fields_out.u**2 + mem_fields_out.v**2)

            # add in the mslp centroid lon/lats for Lagrangian analysis
            mem_track_out = mem_track.loc[mem_track.date.isin(time_intersection)]
            mem_fields_out['centroid_lon'] = ('time', (mem_track_out.lon * 4).round() / 4)
            mem_fields_out['centroid_lat'] = ('time', (mem_track_out.lat * 4).round() / 4)

            # convert to storm frame fields
            mem_fields_out = mem_fields_out.groupby('time').apply(Lagrange.lagrangian_frame)
            mem_fields_out = mem_fields_out.assign(datetime=mem_fields_out.time).drop('time').rename(time='timestep')

            # compute the time of peak vorticity (include moving average to
            # smooth) for storm composites
            peak_vo = mem_track.rolling(3, center=True).mean().vo.idxmax()
            peak_vo_datetime = mem_track.date.loc[peak_vo]
            peak_vo_relative_time = (mem_fields_out.datetime.squeeze().to_pandas() - peak_vo_datetime).dt.total_seconds().values / (3600 * 24)

            # set the storm frame fields timestep relative to peak vorticity time
            mem_fields_out = mem_fields_out.assign_coords(timestep=peak_vo_relative_time)

            LG_fields += [mem_fields_out]

        LG_fields = xr.concat(LG_fields, 'number')
        LG_fields = LG_fields.expand_dims(dict(
            inidate=[pd.to_datetime(inidate)],
            experiment=[exp]))

        return LG_fields

    def preproc_to_stormframe_strongest_deepening(ds,
                                                  ifs_eunice_list=None,
                                                  sfc=True):
        '''
        Funtion for pre-processing to Lagrangian fields for tracked storms at
        their point of strongest deepening.
        Written by Nick Leach and Shirin Ermis.

        Input:
        ------
        ds: xarray dataset
        ifs_eunice_list: pandas dataframe of track information
        sfc: bool, whether surface data or pressure level data is needed
        level:

        Output:
        -------
        LG_fields: xarray dataset with Lagrangian fileds for Eunice like storms
        '''

        ds = ds.copy()

        if 'number' not in ds.coords:
            ds = ds.expand_dims({'number': [0]})

        fpath = ds.encoding['source']
        if sfc:
            exp = fpath.split('/')[-5]
        else:
            exp = fpath.split('/')[-6]
        inidate = fpath.split('/')[-1].split('_')[-1].split('.')[0]
        tmp = 'experiment=="{}" & inidate=="{}"'.format(exp, inidate)
        ds_tracks = ifs_eunice_list.query(tmp)
        LG_fields = []

        unique_num = ds_tracks.number.unique()
        for num in set(ds.number.values).intersection(unique_num):

            mem_track = ds_tracks.loc[ds_tracks.number == num]
            mem_fields = ds.sel(number=num)

            dates = mem_track.date.values
            intersec = set(mem_fields.time.values).intersection(dates)
            time_intersection = sorted(list(intersec))

            resample_freq = 3  # resampling frequency in hours
            if inidate == '2022-02-10':
                resample_freq = 6

            # get start / end times for properly calculating the maximum
            # fields (taking into account the different preproc times in IFS)
            time_delta = pd.Timedelta('{}h 59m'.format(resample_freq - 1))
            time_start = time_intersection[0] - time_delta
            time_end = time_intersection[-1]

            # get the instantaneous fields + wind speeds
            if sfc:
                mem_fields_out = mem_fields.get(['sst',
                                                 'u10',
                                                 'v10',
                                                 'msl',
                                                 'u100',
                                                 'v100',
                                                 'fg10',
                                                 'tcwv'])
                mem_fields_out = mem_fields_out.sel(time=time_intersection)
                mem_fields_out['ws10'] = np.sqrt(mem_fields_out.u10**2 + mem_fields_out.v10**2)
                mem_fields_out['ws100'] = np.sqrt(mem_fields_out.u100**2 + mem_fields_out.v100**2)

                # get the maximum fields, taking into account
                # the different preproc times
                timeslice = slice(time_start, time_end)
                selected_times = mem_fields.mxtpr.sel(time=timeslice)
                mxtpr_field_out = selected_times.resample(time='{}h'.format(resample_freq), label='right', closed='right', base=0).max()
                mem_fields_out['mxtpr'] = mxtpr_field_out

            else:
                mem_fields_out = mem_fields.get(['z',
                                                 'q',
                                                 'r',
                                                 'w',
                                                 't',
                                                 'd',
                                                 'u',
                                                 'v',
                                                 'r',
                                                 'vo']).sel(time=time_intersection)
                mem_fields_out['ws'] = np.sqrt(mem_fields_out.u**2 + mem_fields_out.v**2)

            # add in the mslp centroid lon/lats for Lagrangian analysis
            mem_track_out = mem_track.loc[mem_track.date.isin(time_intersection)]
            mem_fields_out['centroid_lon'] = ('time', (mem_track_out.lon * 4).round() / 4)
            mem_fields_out['centroid_lat'] = ('time', (mem_track_out.lat * 4).round() / 4)

            # convert to storm frame fields
            grouped = mem_fields_out.groupby('time')
            mem_fields_out = grouped.apply(Lagrange.lagrangian_frame)
            mem_fields_out = mem_fields_out.assign(datetime=mem_fields_out.time).drop('time').rename(time='timestep')

            # compute the time of max deepening (include moving average to
            # smooth) for storm composites
            smoothed_msl = mem_track.rolling(3, center=True).mean().msl
            max_deep = smoothed_msl.diff(dim='time').idxmin()
            # TODO: do for gph on pressure levels
            max_deep_datetime = mem_track.date.loc[max_deep]
            max_deep_relative_time = (mem_fields_out.datetime.squeeze().to_pandas() - max_deep_datetime).dt.total_seconds().values / (3600 * 24)

            # set the storm frame fields timestep relative to
            # maximum deepening time
            mem_fields_out = mem_fields_out.assign_coords(timestep=max_deep_relative_time)

            LG_fields += [mem_fields_out]

        LG_fields = xr.concat(LG_fields, 'number')
        LG_fields = LG_fields.expand_dims(dict(
            inidate=[pd.to_datetime(inidate)],
            experiment=[exp]))

        return LG_fields
