"""
Ensemble-mean (opa0-4) ocean temperature delta between the pi-co2 perturbed
NEMO restarts and the operational restarts, used by
notebooks/THESIS_INTRO_ocean_deltas.ipynb.

The restarts are ~6.7 GB gzipped / ~17 GB uncompressed each, so each file is
decompressed to a temporary directory, tn is read and the file is deleted again.

Usage: python ocean_deltas_THESIS_INTRO.py [INIDATE] [TMP_DIR]
"""
import numpy as np
import xarray as xr
import subprocess
import multiprocessing
import os
import sys

INI = sys.argv[1] if len(sys.argv) > 1 else '20220214'
TMP_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/user/{}'.format(os.getuid())
EXPERIMENT = 'pi-co2'
MEMBERS = ['opa{}'.format(i) for i in range(5)]
NPROC = 2  # each worker needs ~17 GB of disk in TMP_DIR

IC_DIR = '/gf5/predict/AWH012_LEACH_NASTORM/IC-PREP/'
PERT_FILE = IC_DIR + 'output/{}/0001/{}/restart/2022/0001_nrt_{}_000000_restart.nc.gz'
ORIG_FILE = IC_DIR + 'restarts/{}/{}/0001_nrt_{}_000000_restart.nc.gz'
MESH_FILE = IC_DIR + 'source/si/mesh_mask.nc'

DATA_DIR = '/gf5/predict/AWH028_ERMIS_BABET/Eunice/'
OUT_FILE = os.path.join(DATA_DIR, 'ocean_delta_tn_{}_{}.nc'.format(EXPERIMENT, INI))


def read_tn(gzfile):
    """Decompress a restart, return tn(z, y, x) and remove the decompressed copy."""
    tmpfile = os.path.join(TMP_DIR, '_'.join(gzfile.split('/')[-6:]).replace('.gz', ''))
    subprocess.run('gzip -dc {} > {}'.format(gzfile, tmpfile), shell=True, check=True)
    try:
        with xr.open_dataset(tmpfile) as ds:
            tn = ds.tn.isel(t=0).load()
    finally:
        os.remove(tmpfile)
    print('read', gzfile, flush=True)
    return tn


def member_delta(member):
    pert = read_tn(PERT_FILE.format(EXPERIMENT, member, INI))
    orig = read_tn(ORIG_FILE.format(INI, member, INI))
    return (pert - orig).astype(np.float32)


if __name__ == '__main__':
    os.makedirs(TMP_DIR, exist_ok=True)
    with multiprocessing.Pool(NPROC) as pool:
        deltas = pool.map(member_delta, MEMBERS)

    delta = xr.concat(deltas, dim='member').assign_coords(member=MEMBERS)

    mesh = xr.open_dataset(MESH_FILE).isel(t=0)
    out = xr.Dataset({'delta_tn': delta.mean('member'),
                      'delta_tn_std': delta.std('member'),
                      'tmask': mesh.tmask,
                      'e1t': mesh.e1t,
                      'e2t': mesh.e2t,
                      'gphit': mesh.gphit,
                      'glamt': mesh.glamt})
    out = out.assign_coords(depth=('z', mesh.gdept_0.values))
    out.delta_tn.attrs = {'long_name': 'ensemble-mean (opa0-4) {} minus operational restart tn'.format(EXPERIMENT),
                          'units': 'K'}
    out.to_netcdf(OUT_FILE)
    print('written', OUT_FILE)
