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
_,outdir,restartdir,opa,perturbation = sys.argv

# DEFINE REQUIRED FUNCTIONS

def eos_insitu(t, s, z):
    '''
    NAME:
        eos_insitu

    DESCRIPTION:
        Python version of in situ density calculation done by NEMO
        routine eos_insitu.f90. Computes the density referenced to
        a specified depth from potential temperature and salinity
        using the Jackett and McDougall (1994) equation of state.
       
    USAGE:
        density = eos_insitu(T,S,p)

    INPUTS:
        T - potential temperature (celsius)
        S - salinity              (psu)
        p - pressure              (dbar)
       
    OUTPUTS
        density - in situ density (kg/m3) - 1000.

    NOTES:
        Original routine returned (rho(t,s,p) - rho0)/rho0.
        This version returns rho(t,s,p). Header for eos_insitu.f90
        included below for reference.

        ***  ROUTINE eos_insitu  ***
       
        ** Purpose :   Compute the in situ density from
        potential temperature and salinity using an equation of state
        defined through the namelist parameter nn_eos. nn_eos = 0
        the in situ density is computed directly as a function of
        potential temperature relative to the surface (the opa t
        variable), salt and pressure (assuming no pressure variation
        along geopotential surfaces, i.e. the pressure p in decibars
        is approximated by the depth in meters.
       
        ** Method  :  
        nn_eos = 0 : Jackett and McDougall (1994) equation of state.
        the in situ density is computed directly as a function of
        potential temperature relative to the surface (the opa t
        variable), salt and pressure (assuming no pressure variation
        along geopotential surfaces, i.e. the pressure p in decibars
        is approximated by the depth in meters.
        rho = eos_insitu(t,s,p)
        with pressure                 p        decibars
        potential temperature         t        deg celsius
        salinity                      s        psu
        reference volumic mass        rau0     kg/m**3
        in situ volumic mass          rho      kg/m**3
       
        Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
        t = 40 deg celcius, s=40 psu
       
        References :   Jackett and McDougall, J. Atmos. Ocean. Tech., 1994
       
    AUTHOR:
        Chris Roberts (hadrr)

    LAST MODIFIED:
        2013-08-15 - created (hadrr)
    '''
    # Convert to double precision
    ptem   = np.double(t)    # potential temperature (celcius)
    psal   = np.double(s)    # salintiy (psu)
    depth  = np.double(z)    # depth (m)
    rau0   = np.double(1035) # volumic mass of reference (kg/m3)
    # Read into eos_insitu.f90 varnames  
    zrau0r = 1 / rau0
    zt     = ptem
    zs     = psal
    zh     = depth
    zsr    = np.sqrt(np.abs(psal))   # square root salinity
    # compute volumic mass pure water at atm pressure
    zr1 = ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt-9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
    # seawater volumic mass atm pressure
    zr2    = ( ( ( 5.3875e-9*zt-8.2467e-7 ) *zt+7.6438e-5 ) *zt-4.0899e-3 ) *zt+0.824493
    zr3    = ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
    zr4    = 4.8314e-4
    #  potential volumic mass (reference to the surface)
    zrhop  = ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1
    # add the compression terms
    ze     = ( -3.508914e-8*zt-1.248266e-8 ) *zt-2.595994e-6
    zbw    = (  1.296821e-6*zt-5.782165e-9 ) *zt+1.045941e-4
    zb     = zbw + ze * zs
    zd     = -2.042967e-2
    zc     =   (-7.267926e-5*zt+2.598241e-3 ) *zt+0.1571896
    zaw    = ( ( 5.939910e-6*zt+2.512549e-3 ) *zt-0.1028859 ) *zt - 4.721788
    za     = ( zd*zsr + zc ) *zs + zaw
    zb1    =   (-0.1909078*zt+7.390729 ) *zt-55.87545
    za1    = ( ( 2.326469e-3*zt+1.553190)*zt-65.00517 ) *zt+1044.077
    zkw    = ( ( (-1.361629e-4*zt-1.852732e-2 ) *zt-30.41638 ) *zt + 2098.925 ) *zt+190925.6
    zk0    = ( zb1*zsr + za1 )*zs + zkw
    # Caculate density
    prd    = (  zrhop / (  1 - zh / ( zk0 - zh * ( za - zh * zb ) )  ) - rau0  ) * zrau0r
    rho    = (prd*rau0) + rau0
    return rho - 1000

# GET DATA

## get land-sea mask
ORCA025Z75_lsm = xr.open_dataset('/network/group/aopp/predict/AWH012_LEACH_NASTORM/IC-PREP/source/si/mesh_mask.nc')

## get list of ORIGINAL restart files
# restarts_3d = glob.glob(restartdir+'*/*restart.nc')

# Use GRAD DESC TO CONSERVE W, U, V FIELDS via SALINITY

# for fpath in restarts_3d:

fpath = glob.glob(restartdir+opa+'/*restart.nc')[0]

## set path variables
# opa = fpath.split('/')[-2]
fname = fpath.split('/')[-1]
expver,nrt,inidate,_,_ = fname.split('_')
print('\n'+opa+':',flush=True)

## get original restart
restart0 = xr.open_dataset(fpath)

## set initial values for EoS
t0 = restart0.tn.values.copy()
s0 = restart0.sn.values.copy()
z = restart0.nav_lev.values.copy()[None,:,None,None]

## find initial density, set delta salinity & tolerance
rho0 = np.where(ORCA025Z75_lsm.tmask,eos_insitu(t0,s0,z),0)
rho_tol = 1e-12
ds = 0.01
    
# for perturbation in ['pi-co2','incr-co2']:
        
restart = xr.open_dataset(outdir+'tmp/'+opa+'-'+perturbation+'-'+fname)

## set initial salinity, temperature values
t = restart.tn.values.copy()
s = s0.copy()

## initial computation of EoS using perturbed temp & original salinity
rho = np.where(ORCA025Z75_lsm.tmask,eos_insitu(t,s,z),0)
rho_diff = rho - rho0

niter=0
while np.any(np.abs(rho_diff) > rho_tol):
    niter+=1
    print(f'interation {niter} max diff:',np.max(np.abs(rho_diff)),flush=True)
    drho_ds = (rho - eos_insitu(t,s+ds,z)) / ds
    s_diff = rho_diff / drho_ds
    s += s_diff
    rho = np.where(ORCA025Z75_lsm.tmask,eos_insitu(t,s,z),0)
    rho_diff = rho - rho0

## set the final salinity field to be all POSITIVE values computed
constrained_salinity = xr.zeros_like(restart0.sn)+np.where(s>=0,s,0)

## set perturbed restart salinity equal to constrained value and save, deleting the "rhop" variable to force interactive computation.
restart['sn'] = constrained_salinity

restart.drop_vars('rhop').to_netcdf(outdir+''+perturbation+'/'+expver+'/'+opa+'/restart/2022/'+fname)

print('saved constrained salinity restart to '+outdir+''+perturbation+'/'+expver+'/'+opa+'/restart/2022/'+fname,flush=True)

restart.close()

restart0.close()