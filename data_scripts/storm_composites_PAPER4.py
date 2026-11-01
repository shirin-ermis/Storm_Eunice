import matplotlib
matplotlib.use('Agg')  # no display on the compute nodes
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import random
import multiprocessing
import glob
import os
import sys
import dask
import metpy.calc
from io import BytesIO
from metpy.units import units
from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

sys.path.insert(0, '/home/e/ermis/Storm_Eunice')  # stormeunice is not pip-installed in this env
import stormeunice as eun

random.seed(10)
dask.config.set(**{'array.slicing.split_large_chunks': True})

style = eun.style

EXPERIMENTS = ['pi', 'curr', 'incr']
INI = '2022-02-14'
TS = 0
PLEV = 850

DATA_DIR = '/gf5/predict/AWH028_ERMIS_BABET/Eunice/'
FIG_DIR = '/home/e/ermis/Storm_Eunice/notebooks/figures'
NOTMISSING = '/home/e/ermis/Storm_Eunice/notebooks/11_members_with_complete_data_in_z_2022-02-14_plev850.npz'

# only the 2022-02-14 initialisation is plotted, so only that file is read
PL_FILES = {'pi': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/pi/EU025/pl/pf/extravars/*_{}.nc',
            'curr': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/ENS/EU025/pl/pf/extravars/*_{}.nc',
            'incr': '/network/group/aopp/predict/AWH012_LEACH_NASTORM/DATA/MED-R/EXP/incr/EU025/pl/pf/extravars/*_{}.nc'}


def get_eunice_tracks():
    '''
    Find the Eunice-like storm in each IFS ensemble member.
    Same filtering as in PAPER4_storm_composites.ipynb.

    Output:
    -------
    ifs_eunice_list: pandas dataframe of track information
    '''
    P1 = multiprocessing.Pool(4)
    ifs_tracks = P1.map(eun.lagrangian.Lagrange.import_medr_tracks_TE,
                        glob.glob('/home/l/leach/Attribution/NA-Storms/Nick/Scripts/tracking/TE/TEStitch_[!2]*'))
    P1.close()

    ifs_tracks = pd.concat(ifs_tracks)
    ifs_tracks['date'] = pd.to_datetime(ifs_tracks.loc[:, ['year', 'month', 'day', 'hour']])

    # ERA5 track
    era_track = pd.read_csv('/home/l/leach/Attribution/NA-Storms/Nick/Scripts/tracking/TE/TEStitch_2022_0',
                            skipinitialspace=True)
    era_track['expid'] = 'era5'
    era_track['experiment'] = 'era5'
    era_track['inidate'] = pd.to_datetime('2022-01-01')
    era_track['number'] = 0
    era_track['date'] = pd.to_datetime(era_track.loc[:, ['year', 'month', 'day', 'hour']])
    eunice_track = era_track.query('track_id==5')

    ifs_tracks_filter = ifs_tracks.groupby(['track_id', 'inidate', 'experiment', 'number']).agg(dict(date=min, day=np.size, lon=lambda x: x.iloc[0], lat=lambda x: x.iloc[0], msl=min))
    eunice_filter = eunice_track.groupby(['track_id', 'inidate', 'experiment', 'number']).agg(dict(date=min, day=np.size, lon=lambda x: x.iloc[0], lat=lambda x: x.iloc[0], msl=min)).iloc[0]

    # starttime criterion - must be detected within 12 hours (before or after) the ERA5 track detection time
    starttime_crit = (ifs_tracks_filter.date - eunice_filter.date).dt.total_seconds().abs() < (12*60*60)
    # starting location criterion - must start within 10 degrees of the ERA5 track starting point
    startloc_crit = np.sqrt((ifs_tracks_filter.lon-eunice_filter.lon)**2+(ifs_tracks_filter.lat-eunice_filter.lat)**2) < 10
    # depth criterion - must reach a depth of at least 980 hPa
    minmsl_crit = ifs_tracks_filter.msl < 98000

    ifs_tracks_filtered = ifs_tracks_filter.loc[starttime_crit & startloc_crit & minmsl_crit]
    ifs_eunice_list = ifs_tracks.set_index(['track_id', 'inidate', 'experiment', 'number']).sort_index().loc[ifs_tracks_filtered.index].reset_index()

    # keep only the track closest to ERA5 in members with more than one candidate
    multi_track_members = ifs_eunice_list.groupby(['inidate', 'experiment', 'number', 'track_id']).agg(dict(year=lambda x: 1)).groupby(['inidate', 'experiment', 'number']).count().query('year > 1')
    for idx in multi_track_members.index:
        track_dists = ifs_eunice_list.query('inidate=="{}" & experiment=="{}" & number=={}'.format(*idx)).groupby('track_id').apply(lambda x: eun.lagrangian.Lagrange.eunice_dist(x, eunice_track=eunice_track))
        num_reject = track_dists.idxmax()
        ifs_eunice_list_idx = ifs_eunice_list.query('inidate=="{}" & experiment=="{}" & number=={} & track_id=={}'.format(*idx, num_reject)).index
        ifs_eunice_list = ifs_eunice_list.drop(ifs_eunice_list_idx)

    return ifs_eunice_list


def calc_storm_composites(ifs_eunice_list, experiment):
    '''
    Storm-frame pressure level fields at the time of maximum deepening
    for all members of one experiment.
    '''
    storms = xr.open_mfdataset(PL_FILES[experiment].format(INI),
                               preprocess=lambda x: eun.lagrangian.Lagrange.preproc_to_stormframe_max_deepening(x, ifs_eunice_list=ifs_eunice_list, sfc=False),
                               combine='nested',
                               concat_dim=['inidate'],
                               chunks={'time': 1, 'number': 1},  # the files are 20-46 GB, avoid one chunk per variable
                               parallel=True,
                               engine='netcdf4')
    storms = storms.sel(inidate=INI, timestep=TS).compute()

    # encodings from the int16 input files do not carry over to the storm frame
    for var in storms.variables:
        storms[var].encoding = {}
    return storms


def plot_q(pl_storms, notmissing, sign_test_q):
    '''
    Specific humidity composites at 850 hPa (panels a-d)
    '''
    # setup -------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.1), sharey=True)
    vmin = 0
    vmax = 5
    lev_number = 10
    z_min = 10000
    z_max = 15000
    num_levels_z = 10
    w_min = -3
    w_max = -0.5
    num_levels_w = 2

    # type sizes are quoted at style.REF_WIDTH and scaled to this canvas, so they
    # print at the same size as the other paper figures once scaled to page width
    fs_title = style.scaled(style.FS_TITLE, fig)
    fs_label = style.scaled(style.FS_LABEL, fig)
    fs_tick = style.scaled(style.FS_TICK, fig)

    # one panel per experiment -------------------------
    composites = {}
    for ax, experiment in zip(axes, EXPERIMENTS):
        comp = (pl_storms[experiment]
                .sel(level=PLEV,
                     number=notmissing['notmissing_' + experiment])
                .squeeze().mean(dim='number'))
        composites[experiment] = comp

        c = (comp.q*1000).plot.contourf(levels=lev_number, cmap=style.CMAP_SEQ, ax=ax,
                                        add_colorbar=False, vmin=vmin, vmax=vmax)

        # geopotential isolines, in white so they clear the dark end of CMAP_SEQ
        cs = ax.contour(comp.storm_lon.values, comp.storm_lat.values, comp.z,
                        np.linspace(z_min, z_max, num_levels_z),
                        colors=style.CONTOUR_LIGHT, linewidths=0.8, zorder=16)
        plt.clabel(cs, fmt='%d', fontsize=fs_tick)

        # isolines for vertical velocities
        ax.contour(comp.storm_lon.values, comp.storm_lat.values, comp.w,
                   np.linspace(w_min, w_max, num_levels_w),
                   colors='limegreen', zorder=19)

        ax.set_title(style.experiment_label(experiment), fontsize=fs_title, fontweight='bold')
        ax.set_xlabel('')  # xarray labels the axes from the variable names
        ax.tick_params(axis='both', which='major', labelsize=fs_tick)

    # diff -------------------------
    diff = composites['incr'] - composites['pi']
    ax = axes[3]
    c_diff = (diff.q*1000).plot.contourf(levels=lev_number, cmap=style.CMAP_DIV, ax=ax,
                                         add_colorbar=False, vmin=-1, vmax=1)

    # isolines, in ink here: the centre of CMAP_DIV is white
    cs = ax.contour(diff.storm_lon.values, diff.storm_lat.values, diff.z,
                    np.linspace(-600, 600, num_levels_z),
                    colors=style.CONTOUR_DARK, linewidths=0.8, zorder=16)
    plt.clabel(cs, fmt='%d', fontsize=fs_tick)

    # hatching for significance
    ax.contourf(diff.storm_lon.values, diff.storm_lat.values, sign_test_q, alpha=0,
                levels=[-0.5, 0.5, 1], hatches=[None, style.HATCH], zorder=21)

    ax.set_title('future –\npreindustrial', fontsize=fs_title, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=fs_tick)

    # axes -------------------------
    for ax in axes:
        ax.set_xticks([-5, 0, 5])
    axes[0].set_ylabel('Latitude from\nstorm centre (°)', fontsize=fs_label)
    for ax in axes[1:]:
        ax.set_ylabel('')

    # y axis number formatting
    axes[0].yaxis.set_major_locator(MultipleLocator(5))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[0].yaxis.set_minor_locator(MultipleLocator(5))

    fig.supxlabel('Longitude from storm centre (°)', x=0.43, y=0.0, va='bottom',
                  fontsize=fs_label)

    # panel labels a, b, c, d
    style.panel_labels(axes)

    # cbars -------------------------
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.2, top=0.8, wspace=0.12)

    cax = fig.add_axes([0.8, 0.2, 0.012, 0.6])  # left, bottom, width, height
    cbar = plt.colorbar(c, cax=cax, format='%.1f')
    cbar.set_label(label='Specific humidity\n(g kg$^{-1}$)', size=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    cax_diff = fig.add_axes([0.91, 0.2, 0.012, 0.6])
    cbar_diff = plt.colorbar(c_diff, cax=cax_diff, format='%.1f')
    cbar_diff.set_label(label='Specific humidity\ndifference (g kg$^{-1}$)', size=fs_label)
    cbar_diff.ax.tick_params(labelsize=fs_tick)
    cbar_diff.locator = MaxNLocator(nbins=5)
    cbar_diff.update_ticks()

    return fig


def plot_vo(pl_storms, pot_temp, sign_test_vo):
    '''
    Vorticity cross-sections through the storm centre (panels e-h)
    '''
    # setup -------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    vmax = 7
    levs = 11
    t_min = 230
    t_max = 360
    num_levels_t = 15

    fs_title = style.scaled(style.FS_TITLE, fig)
    fs_label = style.scaled(style.FS_LABEL, fig)
    fs_tick = style.scaled(style.FS_TICK, fig)

    # one panel per experiment -------------------------
    composites = {}
    for ax, experiment in zip(axes, EXPERIMENTS):
        comp = (pl_storms[experiment].sel(storm_lat=0)
                .squeeze().mean(dim='number'))
        composites[experiment] = comp

        # vorticity is signed about zero, so it takes the signed colourmap rather
        # than the sequential one used for the humidity panels
        c = (comp.vo*10e3).plot.contourf(levels=levs, cmap=style.CMAP_SIGNED, ax=ax,
                                         center=0, vmax=vmax, add_colorbar=False)

        # potential temperature isolines
        cs = ax.contour(comp.storm_lon.values, comp.level.values,
                        pot_temp[experiment].sel(storm_lat=0)
                                            .mean(dim='number').squeeze(),
                        np.linspace(t_min, t_max, num_levels_t),
                        colors=style.CONTOUR_DARK, linewidths=0.8, zorder=16)
        plt.clabel(cs, fmt='%d', fontsize=fs_tick)

        ax.set_title(style.experiment_label(experiment), fontsize=fs_title, fontweight='bold')
        ax.set_xlabel('')  # xarray labels the axes from the variable names
        ax.set_ylim(150, 1000)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax.invert_yaxis()

    # diff -------------------------
    diff = composites['incr'] - composites['pi']
    ax = axes[3]
    c_diff = (diff.vo*10e3).plot.contourf(ax=ax, levels=levs, cmap=style.CMAP_DIV,
                                          vmin=-2, vmax=2, center=0, add_colorbar=False)

    cs = ax.contour(diff.storm_lon.values, diff.level.values,
                    pot_temp['incr'].sel(storm_lat=0).mean(dim='number').squeeze()
                    - pot_temp['pi'].sel(storm_lat=0).mean(dim='number').squeeze(),
                    np.linspace(-5, 5, num_levels_t),
                    colors=style.CONTOUR_DARK, linewidths=0.8, zorder=16)
    plt.clabel(cs, fmt='%d', fontsize=fs_tick)

    # hatching for significance
    ax.contourf(diff.storm_lon.values, diff.level.values, sign_test_vo, alpha=0,
                levels=[-0.5, 0.5, 1], hatches=[None, style.HATCH], zorder=19)

    ax.set_title('future –\npreindustrial', fontsize=fs_title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(150, 1000)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fs_tick)
    ax.invert_yaxis()

    # axes -------------------------
    for ax in axes:
        ax.set_xticks([-5, 0, 5])
    axes[0].set_ylabel('Pressure (hPa)', fontsize=fs_label)
    for ax in axes[1:]:
        ax.set_ylabel('')

    # y axis number formatting: the axis is log-spaced, so it is labelled at the
    # standard pressure levels rather than on a fixed step
    axes[0].set_yticks([200, 300, 500, 700, 850, 1000])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[0].minorticks_off()

    fig.supxlabel('Longitude from storm centre (°)', x=0.43, y=0.0, va='bottom',
                  fontsize=fs_label)

    # panel labels continue the lettering of the humidity figure
    style.panel_labels(axes, letters='efgh')

    # cbars -------------------------
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.19, top=0.81, wspace=0.12)

    cax = fig.add_axes([0.8, 0.19, 0.012, 0.62])  # left, bottom, width, height
    cbar = plt.colorbar(c, cax=cax, format='%.1f')
    cbar.set_label(label='Vorticity\n(10$^{-3}$ s$^{-1}$)', size=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    cax_diff = fig.add_axes([0.91, 0.19, 0.012, 0.62])
    cbar_diff = plt.colorbar(c_diff, cax=cax_diff, format='%.1f')
    cbar_diff.set_label(label='Vorticity difference\n(10$^{-3}$ s$^{-1}$)', size=fs_label)
    cbar_diff.ax.tick_params(labelsize=fs_tick)
    cbar_diff.locator = MaxNLocator(nbins=5)
    cbar_diff.update_ticks()

    return fig


def combine_rows(fig_q, fig_vo, fpath):
    '''
    Stack the humidity (a-d) and vorticity (e-h) figures into one. The vorticity
    row drops its column titles and the humidity row its longitude label, so
    neither is repeated.
    '''
    fig_q._supxlabel.set_visible(False)
    for ax in fig_vo.axes[:4]:
        ax.title.set_visible(False)

    tight = [f.get_tightbbox(f.canvas.get_renderer()).padded(0.1) for f in (fig_q, fig_vo)]
    x0, x1 = min(b.x0 for b in tight), max(b.x1 for b in tight)  # shared, so the panels line up

    rows = []
    for f, b in zip((fig_q, fig_vo), tight):
        buf = BytesIO()
        f.savefig(buf, format='png', dpi=600, bbox_inches=Bbox([[x0, b.y0], [x1, b.y1]]))
        buf.seek(0)
        rows.append(plt.imread(buf))
    width = min(r.shape[1] for r in rows)
    plt.imsave(fpath, np.vstack([r[:, :width] for r in rows]), dpi=600)


if __name__ == '__main__':

    style.set_style()

    # Calculate storm composites ---------------------------
    print('Finding Eunice tracks')
    ifs_eunice_list = get_eunice_tracks()

    pl_storms = {}
    for experiment in EXPERIMENTS:
        print('Calculating storm composites for ' + experiment)
        pl_storms[experiment] = calc_storm_composites(ifs_eunice_list, experiment)
        pl_storms[experiment].to_netcdf(os.path.join(DATA_DIR, 'storm_composites_{}.nc'.format(experiment)))

    # Load members with complete data and significance tests ---------------------------
    notmissing = np.load(NOTMISSING)
    # the bootstraps in the notebook are not seeded, these are the ones
    # behind the published figures
    sign_test_q = np.loadtxt(os.path.join(DATA_DIR, 'sign_test_q.csv'), delimiter=',')
    sign_test_vo = np.loadtxt(os.path.join(DATA_DIR, 'sign_test_vo.csv'), delimiter=',')

    # Potential temperature ---------------------------
    pot_temp = {}
    for experiment in EXPERIMENTS:
        pot_temp[experiment] = metpy.calc.potential_temperature(pl_storms[experiment].level * units.hPa,
                                                                pl_storms[experiment].t * units.kelvin)

    # Plot ---------------------------
    print('Plotting')
    fig_q = plot_q(pl_storms, notmissing, sign_test_q)
    fig_vo = plot_vo(pl_storms, pot_temp, sign_test_vo)
    combine_rows(fig_q, fig_vo, os.path.join(FIG_DIR, 'PAPER4_q_vo.png'))
