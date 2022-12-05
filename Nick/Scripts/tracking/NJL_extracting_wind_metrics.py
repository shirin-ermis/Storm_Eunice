import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import tqdm
import glob
import sys

# IMPORT DATA

_, in_dir, wind_dir, lsm_file = sys.argv[:]

print('importing tracks',flush=True)
track_paths = glob.glob(in_dir+'tracking12_4TestTracks/SystemTracks/*/*.h5')
# get cyclone track data
cyclone_tracks = pd.concat([pd.read_hdf(track_path,'systemtracks') for track_path in track_paths])

# add in date field
# cyclone_tracks['date'] = pd.TimedeltaIndex(cyclone_tracks.time,unit='days') + pd.to_datetime('1949-12-01')

# add in total distance covered field
cyclone_tracks['Ddist'] = np.sqrt((cyclone_tracks.Dx*25)**2+(cyclone_tracks.Dy*25)**2)

# generate unique id using year/mon/sid
cyclone_tracks['uid'] = cyclone_tracks.year.astype(str)+cyclone_tracks.mon.apply(lambda x: '{:02d}'.format(x))+cyclone_tracks.sid.apply(lambda x: '{:03d}'.format(x))

# make date / id index
cyclone_tracks = cyclone_tracks.reset_index(drop=True)

# add in age field
cyclone_tracks['age'] = cyclone_tracks.time - cyclone_tracks.groupby('uid')['time'].transform(np.min)

# get ancillary fields (lsm/cellarea)
## REMEMBER CORDEX sftlf goes from 0->100!
ancil = xr.open_mfdataset(lsm_file).squeeze().load().rename(sftlf='lsm')

# GET METRICS

cyclone_tracks['w6lX'] = np.nan

print('extracting wind metric for each track step',flush=True)
for fpath in sorted(glob.glob(wind_dir+'sfcWind*.nc')):
    
    print('getting wind metrics for {}'.format(fpath),flush=True)
    
    year_data = xr.open_dataset(fpath,decode_times=False)
    ## catch for if unit is hours rather than days as per cyclonetracking
    if year_data.time.units.split(' ')[0] == 'hours':
        year_data = year_data.assign_coords(time=year_data.time/24)
    try:
        year_data = year_data.rename(lat='latitude',lon='longitude')
    except:
        pass
    dim_names = list(year_data.sfcWind.dims)
    dim_names.remove('time')

    year_tracks = cyclone_tracks.loc[cyclone_tracks.time.isin(year_data.time.values)]
    track_fields = year_data.sfcWind.sel(time=year_tracks.time.to_xarray())
    track_centers = year_tracks.loc[:,['lat','long','radius']].to_xarray().rename(lat='latitude',long='longitude')
    track_masks = np.sqrt((year_data.latitude - track_centers.latitude)**2 + (year_data.longitude - track_centers.longitude)**2)
    mask6 = track_masks<6
    track_metric = ((track_fields*mask6) * (ancil.lsm>50)).max(dim_names).to_pandas()
    cyclone_tracks.loc[track_metric.index,'w6lX'] = track_metric.values
    
    year_data.close()
    
# SAVE METRICS

print('saving extracted metrics to {}wind-metrics.h5'.format(in_dir),flush=True)
cyclone_tracks.to_hdf(in_dir+'wind-metrics.h5','wind_metrics')