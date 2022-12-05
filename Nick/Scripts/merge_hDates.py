import xarray as xr
import numpy as np
import pandas as pd
import glob
import sys

_,fname,dirpath=sys.argv

def preproc_mclim(ds):
    ds=ds.copy()
    fname = ds.encoding['source'].split('/')[-1].split('.')[0]
    ds=ds.expand_dims({'hDate':[pd.to_datetime(fname)]})
    ds=ds.transpose('time',...)
    return ds

merge_files = glob.glob(dirpath+'/tmp/*.nc')

print('merging files:\n'+'\n'.join(merge_files),flush=True)

ds=xr.open_mfdataset(merge_files,preprocess=preproc_mclim)

ds=ds.load()

ds.to_netcdf(dirpath+'/'+fname+'.nc')

print('merged to '+dirpath+'/'+fname+'.nc',flush=True)
