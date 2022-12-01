import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import xarray as xr
import pandas as pd
import tqdm
import glob
import sys

_, infile, gridfile, variable, outfile = sys.argv[:]

print('regridding {} {} using {} grid and saving to {}'.format(infile,variable,gridfile,outfile))

ds_in = xr.open_dataset(infile)
grid_out = xr.open_dataset(gridfile)

# convert ds_in longitudes to -180 to 180
if ds_in[[x for x in ds_in.variables if x[:3] == 'lon'][0]].values.max() > 180:
    print('converting longitudes to -180 to 180 scale')
    ds_in = ds_in.assign_coords({[x for x in ds_in.variables if x[:3] == 'lon'][0]:(ds_in[[x for x in ds_in.variables if x[:3] == 'lon'][0]]+180)%360-180})

# Get lat/lon values (accounting for latitude / lat naming)
in_lat = ds_in[[x for x in ds_in.variables if x[:3] == 'lat'][0]].values
in_lon = ds_in[[x for x in ds_in.variables if x[:3] == 'lon'][0]].values
in_vals = ds_in[variable].values

grid_lat = grid_out[[x for x in grid_out.variables if x[:3] == 'lat'][0]].values
grid_lon = grid_out[[x for x in grid_out.variables if x[:3] == 'lon'][0]].values

# if lat/lon on a regular grid, convert to mesh
if np.ndim(in_lat) == 1:
    in_lon,in_lat = np.meshgrid(in_lon,in_lat)

# Precompute Delaunay triangulation
tri = Delaunay(np.array([in_lon.flatten(),in_lat.flatten()]).T)

# Slice EASE2 grid to minimal bbox
interp = LinearNDInterpolator(tri, in_vals[0].flatten()*0)
minbox = np.nan_to_num(interp(grid_lon,grid_lat)+1).astype(bool)
U,V = np.meshgrid(grid_out.u,grid_out.v)
U_min = U[minbox].min()
U_max = U[minbox].max()
V_min = V[minbox].min()
V_max = V[minbox].max()

# generate new subgrid lat/lon mesh
subgrid_out = grid_out.sel(u=slice(U_min,U_max),v=slice(V_max,V_min))
subgrid_lat = subgrid_out[[x for x in grid_out.variables if x[:3] == 'lat'][0]].values
subgrid_lon = subgrid_out[[x for x in grid_out.variables if x[:3] == 'lon'][0]].values

# create empty array for regridded data
interp_vals = np.empty((in_vals.shape[0],*subgrid_out.z.shape))

# loop over first input index
for i,fld in tqdm.tqdm(enumerate(in_vals)):
    
    interp = LinearNDInterpolator(tri, fld.flatten())
    interp_vals[i] = interp(subgrid_lon,subgrid_lat)
    
# retain attributes from the original, but add in history line
update_attrs = ds_in.attrs.copy()
if 'history' in update_attrs:
    update_attrs['history'] = pd.to_datetime('today').strftime('%c')+': Regridded using scipy LinearNDInterpolator to EASE2.0 25km EU grid \n'+update_attrs['history']
else:
    update_attrs['history'] = pd.to_datetime('today').strftime('%c')+': Regridded using scipy LinearNDInterpolator to EASE2.0 25km EU grid'

# create Dataset & write to netcdf
regridded = xr.Dataset(data_vars = {variable:(('time','v','u'),interp_vals),'lat':subgrid_out[[x for x in grid_out.variables if x[:3] == 'lat'][0]],'lon':subgrid_out[[x for x in grid_out.variables if x[:3] == 'lon'][0]]},
                       coords = {'time':ds_in.time,'u':subgrid_out.u,'v':subgrid_out.v},
                       attrs = update_attrs)

# if only wanting 3-hrly (or coarser) data:
regridded_mod3 = regridded.time.dt.hour.values[0]%3
regridded = regridded.sel(time=regridded.time.dt.hour%3==regridded_mod3)

regridded.to_netcdf(outfile)

# also write the subgrid to a file for elevation
subgrid_out.to_netcdf('/'.join(outfile.split('/')[:-1])+'/elevation_grid.nc')

print('done')
