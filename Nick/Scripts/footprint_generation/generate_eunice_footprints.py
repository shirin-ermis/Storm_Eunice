import rioxarray as rio
import xarray as xr
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy import ndimage
import glob
import gc

import dask
from dask.distributed import Client, LocalCluster

def setUpCluster(
    n_workers: int, low_workers: int, high_workers: int, memory_limit: int
):
    dask.config.set({"temporary_directory": "/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/tmp/"})
    dask.config.set({"array.slicing.split_large_chunks": True})
    dask.config.set({"distributed.worker.memory.spill": 0.8})
    dask.config.set({"distributed.worker.use_file_locking": True})
    # DASK CLUSTER SET UP
    cluster = LocalCluster(
        n_workers=n_workers,
        dashboard_address="localhost:14286",
        memory_limit=f"{memory_limit} GiB",
        threads_per_worker=2,
    )
    cluster.adapt(minimum=low_workers, maximum=high_workers)

    print(f"dashboard : {cluster.dashboard_link}")
    client = Client(cluster)

    return cluster, client

## gust parameterisation function
def gust_parameterisation(ds):
    
    u10 = np.sqrt(ds.u10**2+ds.v10**2)
    u100 = np.sqrt(ds.u100**2+ds.v100**2)
    
    wgSLH = u10 + 3.25 * (u100 - u10) / np.log(100/10)
    
    wg_out = 10.3 + 0.0112*ds.fg10**2 + 0.0148 * wgSLH**2 + 0.00355 * ds.dem
    
    return wg_out.rename('gusts_parameterised')

## precompute interp func
ds = xr.open_dataset('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/sfc/cf/1_2022-02-16.nc')
ds = ds.sel(longitude=slice(-12,4),latitude=slice(63,49))
lons = ds.longitude.values
lats = ds.latitude.values

points = np.array(np.meshgrid(lons,lats)).reshape(2,-1).swapaxes(0,1)

tri = Delaunay(points)

def interp_vals(A, x, y):
    # print(A.shape)

    points_out = np.array(np.meshgrid(x,y)).reshape(2,-1).swapaxes(0,1)

    interp = LinearNDInterpolator(tri,A.flatten())
    
    out_image = interp(points_out).reshape(y.size,x.size)
    invalid = np.isnan(out_image)
    ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    out_image_fill = out_image[tuple(ind)]

    return out_image_fill

ds.close()

def preprocess_and_save(fname,dem):
    
    name = fname.split('/')[-1].split('.')[0]
        
    print('computing footprints for {}'.format(fname),flush=True)

    ds = xr.open_dataset(fname, chunks={'time':1})
    ds = ds.sel(time=slice('2022-02-17','2022-02-19 23'),longitude=slice(-12,4),latitude=slice(63,49))

    if not 'number' in ds.coords:
        ds = ds.expand_dims({'number':[0]})

    for mem in ds.number.values:

        print('computing footprints for member {}'.format(mem),flush=True)

        ds_remap = xr.apply_ufunc(
            interp_vals,
            ds.sel(number=mem),
            dem.x.values,
            dem.y.values,
            input_core_dims=[['latitude','longitude'],['x'],['y']],
            output_core_dims=[['y','x']],
            exclude_dims=set(('latitude','longitude'),),
            vectorize=True,
            dask='parallelized',
            output_dtypes = np.float32
        )

        ds_remap['x'] = dem.x.values
        ds_remap['y'] = dem.y.values
        ds_remap['dem'] = (('y','x'), dem.squeeze().values)

        ds_footprint = gust_parameterisation(ds_remap).max('time').squeeze()

        ds_footprint.rio.to_raster('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/WISC/data/event_set_tif/{}_mem{}.tif'.format(name,mem),compute=True)
        
    ds.close()


if __name__ == '__main__':
    
    cluster, client = setUpCluster(n_workers= 4, low_workers= 3, high_workers= 5, memory_limit= 10)
    
    ## import dem:
    dem = rio.open_rasterio('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/ancil/eu_dem_v11_1km.tif')
    dem = xr.where(dem<-1e20,0,dem)
    dem = dem.sel(x=slice(-12,4),y=slice(63,49))
    
#    fnames = glob.glob('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/sfc/*/*.nc')
    fnames = glob.glob('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr/EU025/sfc/*/b2nt*.nc')    
    # fnames = glob.glob('/gf3/predict2/AWH012_LEACH_NASTORM/DATA/ERA5/EU025/sfc/2022.nc')    
    
    for fname in fnames:
        
        preprocess_and_save(fname,dem)
        gc.collect()
