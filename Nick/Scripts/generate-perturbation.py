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

# GET DATA

## World Ocean Atlas 18 (1 degree)
def preproc_WOA18(ds):
    fname = ds.encoding['source']
    fyear = fname.split('/')[-1].split('_')[1]
    year = ('19'+fyear[:2]).replace('19A','200')
    ds['time'] = [pd.to_datetime(year)]
    return ds
WOA18 = xr.open_mfdataset(glob.glob('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/t3d/*t02*.nc'),decode_times=False,preprocess=preproc_WOA18)
### land-sea mask creation
WOA_lsm = pd.read_csv('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/t3d/landsea_01.msk',skiprows=1).set_index(['Latitude','Longitude']).to_xarray().Bottom_Standard_Level
WOA_lsm = xr.concat([(WOA_lsm>i+1).expand_dims({'depth':[x]}) for i,x in enumerate(WOA18.depth)],dim='depth')
WOA_lsm = WOA_lsm.rename(dict(Latitude='lat',Longitude='lon'))

## HadISST
HadISST = xr.open_dataset(cdo.setmisstonn(input='-setctomiss,-1000 -selmonth,2 -selyear,1870/2021 /network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/sst/HadISST_sst.nc'))
lsm_hadisst=~xr.ufuncs.isnan(xr.open_dataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/sst/HadISST_sst.nc').isel(time=-1).sst)

## ORAS5 (for iicethic & ileadfra)
ORAS5_ileadfra = xr.open_mfdataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/si/ileadfra/ileadfra_control_monthly_highres_2D_*02_*_v0.1.nc')
ORAS5_iicethic = xr.open_mfdataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/si/iicethic/iicethic_control_monthly_highres_2D_*02_*_v0.1.nc')
ORCA025Z75_lsm = xr.open_dataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/si/mesh_mask.nc')

## HadCRUT5
HC5 = xr.open_dataset('../Ancil/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.nc')
HC5 = HC5.tas_mean.to_pandas()
HC5.index = HC5.index.year

## ERF components from AR6
erf_ar6 = pd.read_csv('../Ancil/AR6_ERF_1750-2019.csv',index_col=0)
erf_ar6.loc[:,'ghg'] = erf_ar6.loc[:,'total_anthropogenic'] - erf_ar6.loc[:,'aerosol']
### extend ERF to 2022
ssp245_erf = pd.read_csv('../Ancil/ERF_ssp245_1750-2500.csv',index_col=0)
ssp245_erf['aerosol'] = ssp245_erf.loc[:,'aerosol-radiation_interactions'] + ssp245_erf.loc[:,'aerosol-cloud_interactions']
ssp245_erf.loc[:,'ghg'] = ssp245_erf.loc[:,'total_anthropogenic'] - ssp245_erf.loc[:,'aerosol']
for year in [2020,2021,2022]:
    erf_ar6.loc[year] = ssp245_erf.loc[year] * erf_ar6.loc[year-1] / ssp245_erf.loc[year-1]

print('got data',flush=True)

# COMPUTE AWI:

## ant / nat FaIR run
fair_erf = pd.DataFrame(index=erf_ar6.index,columns=pd.MultiIndex.from_product([['ant','aer','nat'],['forcing']]),data=pd.concat([erf_ar6.loc[:,'ghg'],erf_ar6.loc[:,'aerosol'],erf_ar6.loc[:,'total_natural']],axis=1).values)
fair_emms = return_empty_emissions(start_year=1750,end_year=2022,scen_names=['ant','aer','nat'])
fair_temps = run_FaIR(emissions_in=fair_emms,forcing_in=fair_erf)['T'].loc[1850:]

## regress HadCRUT5 onto FaIR temperature output & define anthropogenic warming index
X = np.column_stack([np.ones(fair_temps.loc[:2021].index.size),fair_temps.loc[:2021]])
Y = HC5.loc[1850:2021].values[:,None]
mlr = OLSE.multiple(Y)
mlr.fit(X)
AWI = ( mlr.B[1]*fair_temps.aer + mlr.B[2]*fair_temps.ant ).default

print('computed AWI',flush=True)

# 3D ATTRIBUTABLE WARMING:

## create regressor for WOA data:
timeslices = [slice(x,x+9) for x in np.arange(1955,2005,10)]+[slice(2005,2017)]
WOA18_X = np.array([AWI.loc[timeslice].mean() for timeslice in timeslices])
X = WOA18_X[:,None,None,None]
Y = WOA18.t_an.squeeze().values
olsreg = OLSE.simple( Y = Y )
olsreg.fit( X = X )
### compute estimated attributable warming over 1850-1900 to 2022 period
t3d = olsreg.b1 * (AWI.loc[2022] - AWI.loc[1850:1900].mean())
### create DataArray object
t3d = xr.zeros_like(WOA18.t_an.isel(time=0).squeeze()) + t3d

print('computed attributable t3d',flush=True)

## remap & save 3d field
### save attributable warming on original grid
t3d.where(WOA_lsm).to_netcdf(outdir+'tmp/t3d_raw.nc')
### file with grid to remap onto
horz_grid = '/network/group/aopp/predict/AWH009_LEACH_RELIABLE/COUNTERFACTUAL-FORECASTING/IC-prep/preproc_ocean5/votemper_1m_y2021m06.nc'
### fill in missing values using IDW
t3d_infill = cdo.setmisstoc('0',input='-setmisstodis,8 '+outdir+'tmp/t3d_raw.nc')
### generate weights file for regridding
horz_wts = cdo.genbil(horz_grid,input=t3d_infill)
### remap horizontally
t3d_remaph = cdo.remap(horz_grid+','+horz_wts,input=t3d_infill)
### remap vertically 
#### get the depths of the levels for ORCA025Z72
levels_72 = ','.join(str(x) for x in xr.open_dataset('/network/group/aopp/predict/AWH009_LEACH_RELIABLE/COUNTERFACTUAL-FORECASTING/IC-prep/preproc_ocean5/votemper_1m_y2021m06.nc').deptht.values)
#### interpolate onto 72 levels & save
cdo.intlevelx(levels_72,input=t3d_remaph,output=outdir+'tmp/t3d_remap.nc')

print('infilled & remapped t3d',flush=True)

# 2D ATTRIBUTABLE SST

Y = HadISST.sst.squeeze().values
X = AWI.loc[1870:2021].values[:,None,None]
olsreg = OLSE.simple( Y = Y )
olsreg.fit( X = X )
t2d = olsreg.b1 * (AWI.loc[2022] - AWI.loc[1850:1900].mean())
t2d = xr.zeros_like(HadISST.sst.isel(time=0).squeeze()) + t2d

print('computed attributable sst',flush=True)

## remap & save 2d field
### save attributable warming on original grid
t2d.where(lsm_hadisst).to_netcdf(outdir+'tmp/t2d_raw.nc')
### fill in missing values using IDW
t2d_infill = cdo.setmisstoc('0',input='-setmisstodis,8 '+outdir+'tmp/t2d_raw.nc')
### remap horizontally & save
cdo.remapbil(horz_grid,input=t2d_infill,output=outdir+'tmp/t2d_remap.nc')

print('infilled & remapped sst',flush=True)

# RELAX t3d TO t2d AT SURFACE

## import generated perturbations
t2d_remap = xr.open_dataset(outdir+'tmp/t2d_remap.nc').sst
t3d_remap = xr.open_dataset(outdir+'tmp/t3d_remap.nc').t_an
## create relaxation function (value of 0.2 at 100m depth)
relax_function = np.exp( ( t3d_remap.depth - t3d_remap.depth.isel(depth=0) ) / ( 100 / np.log(0.2) ) )
## calculate WOA18 tendencies from surface
t3d_tendencies = t3d_remap - t3d_remap.isel(depth=0)
## relax between HadISST + WOA18 tendencies -> pure WOA18
t3d_relaxed = (t2d_remap + t3d_tendencies) * relax_function + t3d_remap * (1 - relax_function)
## save this "hybrid" delta (making sure dimension order correct)
t3d_relaxed.transpose('depth','y','x').to_netcdf(outdir+'tmp/t3d_hybrid.nc')

print('relaxed t3d to sst at surface & saved',flush=True)

# 2D SIC & SIT

## compute regression as above
Y0 = ORAS5_ileadfra.ileadfra.squeeze().values
Y1 = ORAS5_iicethic.iicethic.squeeze().values
X = AWI.loc[1958:2021].values[:,None,None]
olsreg0 = OLSE.simple( Y = Y0 )
olsreg1 = OLSE.simple( Y = Y1 )
olsreg0.fit( X = X )
olsreg1.fit( X = X )
sic = olsreg0.b1 * (AWI.loc[2022] - AWI.loc[1850:1900].mean())
sit = olsreg1.b1 * (AWI.loc[2022] - AWI.loc[1850:1900].mean())
sic = xr.zeros_like(ORAS5_ileadfra.ileadfra.isel(time_counter=0).squeeze()) + sic
sit = xr.zeros_like(ORAS5_iicethic.iicethic.isel(time_counter=0).squeeze()) + sit

print('computed attributable sea-ice concentration & thickness',flush=True)

## save these perturbations to disk (no remap needed as ORAS5 grid identical)
sic.where(ORCA025Z75_lsm.tmask.isel(z=0).squeeze()).to_netcdf(outdir+'tmp/sic_raw.nc')
sit.where(ORCA025Z75_lsm.tmask.isel(z=0).squeeze()).to_netcdf(outdir+'tmp/sit_raw.nc')

## infill to avoid land-sea mask issues (not that there should be any...), using common naming structure
cdo.setmisstoc('0',input='-setmisstodis,8 '+outdir+'tmp/sic_raw.nc',output=outdir+'tmp/sic_remap.nc')
cdo.setmisstoc('0',input='-setmisstodis,8 '+outdir+'tmp/sit_raw.nc',output=outdir+'tmp/sit_remap.nc')

print('infilled sea-ice concentration & thickness + saved',flush=True)