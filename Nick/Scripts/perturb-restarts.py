# DEPENDENCIES
import numpy as np
import xarray as xr
import pandas as pd
import glob
import multiprocessing
import tqdm
import sys
from cdo import Cdo
cdo = Cdo()

## get my stats functions
from mystatsfunctions import OLSE,LMoments
from moarpalettes import get_palette

## get fair
from fair import *

# SET ARGUMENTS
outdir = sys.argv[1]
restartdir = sys.argv[2]

# GET DELTA FIELDS
## NB These are pre-industrial -> present deltas
delta_t3d = xr.open_dataarray(outdir+'tmp/t3d_hybrid.nc')
delta_sic = xr.open_dataarray(outdir+'tmp/sic_remap.nc')
delta_sit = xr.open_dataarray(outdir+'tmp/sit_remap.nc')

## get land-sea mask
ORCA025Z75_lsm = xr.open_dataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/si/mesh_mask.nc')

# PERTURB 3D FIELDS
restarts_3d = glob.glob(restartdir+'*/*restart.nc')

for fpath in restarts_3d:
    opa = fpath.split('/')[-2]
    fname = fpath.split('/')[-1]
    print('perturbing '+opa+' t3d',flush=True)
    ## pi-CO2 perturbation first
    restart = xr.open_dataset(fpath)
    ## subtract attributed delta, setting masked locations to zero in the perturbation file
    restart['tn'] -= np.where(ORCA025Z75_lsm.squeeze().tmask.values,delta_t3d.values,0)
    ## save perturbed restart
    restart.to_netcdf(outdir+'tmp/'+opa+'-pi-co2-'+fname)
    restart.close()
   
    ## equivalent perturbation for incr-CO2
    restart = xr.open_dataset(fpath)
    restart['tn'] += np.where(ORCA025Z75_lsm.squeeze().tmask.values,delta_t3d.values,0)
    restart.to_netcdf(outdir+'tmp/'+opa+'-incr-co2-'+fname)
    restart.close()
    
# PERTURB 2D FIELDS
restarts_2d = glob.glob(restartdir+'*/*restart_ice.nc')

def chck_frld(restart):
    ## set values lower than zero to 0
    restart['frld'] -= (restart['frld']<0) * restart['frld']
    ## set values greater than 1 to 1
    restart['frld'] -= (restart['frld']>1) * (restart['frld']-1)
    return restart

def chck_hicif(restart):
    ## set values lower than zero to 0
    restart['hicif'] -= (restart['hicif']<0) * restart['hicif']
    return restart

def chck_phys(restart):
    ## set hicif values where frld is 1 (ie. no ice) to 0
    restart['hicif'] -= (restart['frld']==1) * restart['hicif']
    ## set frld values where hicif is 0 (ie. no ice) to 1
    restart['frld'] -= (restart['hicif']==0) * (restart['frld']-1)
    return restart

for fpath in restarts_2d:
    opa = fpath.split('/')[-2]
    fname = fpath.split('/')[-1]
    print('perturbing '+opa+' si2d',flush=True)
    ## pi-CO2 perturbation first
    restart = xr.open_dataset(fpath)
    ## remove attributed delta (equivalent to add for ileadfrac since frld == 1 - siconc)
    restart['frld'] += np.where(ORCA025Z75_lsm.squeeze().tmask.isel(z=0).values,delta_sic.values,0)
    ## check values are physical
    restart = chck_frld(restart)
    ## remove attributed delta
    restart['hicif'] -= np.where(ORCA025Z75_lsm.squeeze().tmask.isel(z=0).values,delta_sit.values,0)
    ## check values are physical
    restart = chck_hicif(restart)
    restart = chck_phys(restart)
    ## save perturbed restart
    restart = restart.load()
    restart.to_netcdf(outdir+'pi-co2/0001/'+opa+'/restart/2022/'+fname)
    restart.close()
    
    ## equivalent perturbation for incr-CO2
    restart = xr.open_dataset(fpath)
    restart['frld'] -= np.where(ORCA025Z75_lsm.squeeze().tmask.isel(z=0).values,delta_sic.values,0)
    restart = chck_frld(restart)
    restart['hicif'] += np.where(ORCA025Z75_lsm.squeeze().tmask.isel(z=0).values,delta_sit.values,0)
    restart = chck_hicif(restart)
    restart = chck_phys(restart)
    restart = restart.load()
    restart.to_netcdf(outdir+'incr-co2/0001/'+opa+'/restart/2022/'+fname)
    restart.close()