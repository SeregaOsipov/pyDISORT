import os.path
import numpy as np
import pandas as pd
import xarray
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import hvplot.xarray # noqa
import hvplot.pandas # noqa
from climpy.utils.atmos_chem_utils import get_atmospheric_gases_composition
from climpy.utils.atmos_utils import get_atmospheric_profile
from climpy.utils.file_path_utils import make_dir_for_the_full_file_path
from climpy.utils.optics_utils import derive_rayleigh_optical_properties, mix_optical_properties
from climpy.utils.plotting_utils import save_figure
from climpy.utils.lblrtm_utils import Gas, run_lblrtm_over_spectral_range, LblrtmSetup
from climpy.utils.disort_utils import DisortSetup, run_disort_spectral, setup_viewing_geomtry, \
    setup_surface_albedo, RRTM_SW_LW_WN_RANGE, RRTM_LW_WN_RANGE, checkin_and_fix, RRTM_SW_WN_RANGE
from metpy.units import units
from metpy.calc import mixing_ratio_from_relative_humidity

from disort_plotting_utils import plot_spectral_profiles, plot_ref_perturbed_pmc
import scipy as sp
#%% experimental setup
lats = np.array([-20.55,])  # location & date to process
lons = np.array([175.385,])
date = dt.datetime(1991, 1, 2)

SPECTRAL_WN_RANGE = RRTM_SW_LW_WN_RANGE
#%% setup atmospheric meteorological profile
atm_stag_ds = get_atmospheric_profile(date)  # stag indicates staggered profile
atm_stag_ds = atm_stag_ds.sel(lat=lats, lon=lons, time=date, method='nearest')  # TODO: date selection should be controlled by tollerance
atm_stag_ds['p'] = (('level', 'lat', 'lon'), atm_stag_ds['p'].data[:, np.newaxis, np.newaxis])  # keep in 1D although general approach is N-D

# add few z levels for accurate setup of the P sim
    # f = sp.interpolate.interp1d(atm_stag_ds.z.squeeze(), np.log10(atm_stag_ds.level))  # interpolation in log p is Good, but keep it consistent with xarry.interp. Error in strat is not to bad
    # new_p_levels = 10**(f(np.array([18, 23, 28])*10**3))  # hPa
f = sp.interpolate.interp1d(atm_stag_ds.z.squeeze(), atm_stag_ds.level)  # interpolate linear
new_p_levels = f(np.array([18, 23, 25, 28])*10**3)  # hPa  # profile id 3 & 4
new_p_levels = f(np.array([20, 22.5, 25, 28])*10**3)  # hPa  # prodilfe id 5
new_levels = np.sort(np.concatenate((atm_stag_ds.level.data, new_p_levels)))[::-1]
atm_stag_ds = atm_stag_ds.interp(level=new_levels)
#%% setup atmospheric chemical composition (gases). Setup H2O from atm grid (relative humidity)
gases_ds = get_atmospheric_gases_composition()
gases_ds = gases_ds.sel(lat=lats, lon=lons, time=date.month-1, method='nearest')  # do the selection
# interpolate gases on the LBLRTM vertical grid
gases_ds = gases_ds.interp(level=atm_stag_ds.level.data, kwargs={"fill_value": "extrapolate"})  # TODO: extrapolating blindly is bad
# Add H2O, already on the MERRA2 grid
h2o_ds = gases_ds.sel(species=Gas.O2.value).copy(deep=True)  # Use O2 to make a single gas copy as a template for SO2


def rh_to_vmr(atm_stag_ds):
    h2o_mmr = mixing_ratio_from_relative_humidity(atm_stag_ds.p.values * units.hPa, atm_stag_ds.t.values * units.degK, atm_stag_ds.r.values / 100)  # .to('g/kg')
    h20_vmr = 29 / 18 * h2o_mmr

    return h20_vmr


# relative humidity
# h2o_ds.const[:] = atm_stag_ds.r[:].data  # units H
# h2o_ds.const.attrs['units'] = 'relative humidity (%)'
# h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
# h2o_ds['units'] = (('species',), ['H',])  # relative humidity

# mass density
# h2o_ds.const[:] = rel_humidity_to_mass_concentration(atm_stag_ds) * 10**3  # A-ppmv, D-mass density (gm m-3)
# h2o_ds.const.attrs['units'] = 'mass density (gm m-3)'
# h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
# h2o_ds['units'] = (('species',), ['D',])  # mass density (gm m-3)

# volume mixing ratio
h2o_ds.const[:] = rh_to_vmr(atm_stag_ds) * 10**6  # units A
if (h2o_ds.const<0).any():  # replace any negative values
    temp_ds = h2o_ds.sortby('level')
    temp_ds['const'] = temp_ds.const.where(temp_ds.const>0).interpolate_na(dim='level')
    h2o_ds = temp_ds.sortby('level', ascending=False)
    plt.figure()
    h2o_ds.const.plot(yscale='log', xscale='log')

h2o_ds.const.attrs['units'] = 'volume mixing ratio (ppmv)'
h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
h2o_ds['units'] = (('species',), ['A',])  # volume mixing ratio (ppmv)

gases_ds = xr.concat([gases_ds, h2o_ds], 'species')
gases_ds_ref = gases_ds
#%% adjust H2O for reference & perturbed cases
'''
Perturbation sims:
+1 ppmv or + 10 ppmv 40-75 hPa / 4 ppmv background
+1 ppmv or + 10 ppmv / 10-45 hPa 
уменьшить температуру на 2 и 4 градуса в этих слоях

profile_id options:
1: 75-40 hPa
2: 45-10 hPa
3 & 4. step profiles to mimic figures in Sergey Haikin's paper
4. Ukhov's average WRF profile in Feb. +4 ppmv between 25 and 28 km
temperature_id options:
0: +0 K
1: -2 K
2: -4 K
4. Ukhov's average WRF profile in Feb. +0.2 K between 20 and 22.5 km and -0.2K between 25 and 28 km

scenario_id = 'v{profile_id}.{temperature_id}'
'''

profile_id = 5
temperature_id = 4
scenario_label = 'v{}.{}'.format(profile_id, temperature_id)   # first latter is water perturbation, second number is temperature

dvout = 10  # cm^-1 (LBLRTM DVOUT, wn grid)
intermidiate_data_storage_path = '/Users/osipovs/Data/Tonga/dvout_{}/'.format(dvout)

# pics_folder = get_pictures_root_folder()+'Tonga/{}/'.format(scenario_label)
pics_folder = os.path.expanduser('~')+'/Pictures/Tonga/{}/'.format(scenario_label)

levels_slice = slice(75, 40)
if profile_id == 2:
    levels_slice = slice(45, 10)
temperature_perturbation = 0
if temperature_id == 1:
    temperature_perturbation = -2
if temperature_id == 2:
    temperature_perturbation = -4

print('Scenario {}: levels are {}, temperature perturbation is +{} K'.format(scenario_label, levels_slice, temperature_perturbation))

# set RH to 50% in ref and 100% in Tonga sims
# h2o_ds = gases_ds_ref.sel(species=Gas.H2O.value)
# h2o_ds.sel(level=levels_slice)['const'][:] = 50

gases_ds_tonga = gases_ds_ref.copy(deep=True)
h2o_ds = gases_ds_tonga.sel(species=Gas.H2O.value)

p_levels = h2o_ds.level
if profile_id < 3:
    h2o_ds.const.loc[dict(level=levels_slice)]['const'][:] += 10  # ppmv
# elif profile_id == 3:
#     h2o_ds.const.loc[dict(level=p_levels[(p_levels <= new_p_levels[0]) & (p_levels > new_p_levels[1])])] += 0.5  # ppmv
#     h2o_ds.const.loc[dict(level=p_levels[(p_levels <= new_p_levels[1]) & (p_levels > new_p_levels[2])])] += 4
#     h2o_ds.const.loc[dict(level=p_levels[p_levels <= new_p_levels[2]])] += 1
# elif profile_id == 4:
#     h2o_ds.const.loc[dict(level=p_levels[(p_levels <= new_p_levels[0]) & (p_levels > new_p_levels[1])])] += 0.5  # ppmv
#     h2o_ds.const.loc[dict(level=p_levels[(p_levels <= new_p_levels[1]) & (p_levels > new_p_levels[2])])] += 8
#     h2o_ds.const.loc[dict(level=p_levels[p_levels <= new_p_levels[2]])] += 1
elif profile_id == 5:
    h2o_ds.const.loc[dict(level=p_levels[(p_levels <= new_p_levels[2]) & (p_levels >= new_p_levels[3])])] += 4  # ppmv
# levels = h2o_ds['const'].level
# h2o_ds['const'].loc[dict(level=levels[levels<100])] *= 1.1  # boost water vapor above 100 hpa by 10%

diff_ds = gases_ds_tonga.sel(species=Gas.H2O.value).const-gases_ds_ref.sel(species=Gas.H2O.value).const

atm_stag_ds_tonga = atm_stag_ds.copy(deep=True)
if temperature_id < 3:
    atm_stag_ds_tonga.sel(level=levels_slice)['t'][:] += temperature_perturbation
elif temperature_id == 4:
    atm_stag_ds_tonga['t'].loc[dict(level=p_levels[(p_levels <= new_p_levels[0]) & (p_levels >= new_p_levels[1])])] += 0.2
    atm_stag_ds_tonga['t'].loc[dict(level=p_levels[(p_levels <= new_p_levels[2]) & (p_levels >= new_p_levels[3])])] -= 0.2
#%% estimate LNFL & LBLRTM spectral requirements
'''
LNLFL: The TAPE3 file should include lines from at least 25 cm-1 on either end of the calculation region.
'''
print('LNFL TAPE5 range for SW should be: {} {}'.format(SPECTRAL_WN_RANGE[0] - 25, SPECTRAL_WN_RANGE[1] + 25))
#%% LBL: derive gases absorption OD & optical properties
cross_sections = gases_ds.sel(species=[Gas.NO2.value, ])  # indicate the gases, for which the xsections should be accounted for
# cross_sections = gases_ds.sel(species=[Gas.NO2.value, Gas.SO2.value])  # indicate the gases, for which the xsections should be accounted for
lblrtm_scratch_fp = '/Users/osipovs/Temp/'  # local
lblrtm_scratch_fp = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/run_tonga/'  # remote

lblrtm_setup_vo = LblrtmSetup()
lblrtm_setup_vo.DVOUT = dvout  # cm^-1
lblrtm_setup_vo.zenithAngle = 180  # 0

fp_wo = intermidiate_data_storage_path + '/lblrtm_ds_wo_rayleigh.nc'
fp_w = intermidiate_data_storage_path + '/lblrtm_ds_w_rayleigh.nc'
if os.path.exists(fp_w):
    print('Reusing local copy of the Ref LBLRTM calculations')
    lbl_op_ds_wo_rayleigh = xarray.open_dataset(fp_wo)
    lbl_op_ds_w_rayleigh = xarray.open_dataset(fp_w)
else:
    #  Remember that LBLRTM represent rho layer and optical properties of these layers
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    lbl_op_ds_wo_rayleigh = run_lblrtm_over_spectral_range(SPECTRAL_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, lblrtm_setup_vo, False)
    make_dir_for_the_full_file_path(fp_wo)
    lbl_op_ds_wo_rayleigh.to_netcdf(fp_wo)
    print('\nPreparing LBLRTM run with Rayleigh scattering\n')
    lbl_op_ds_w_rayleigh = run_lblrtm_over_spectral_range(SPECTRAL_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, lblrtm_setup_vo, True)
    make_dir_for_the_full_file_path(fp_w)
    lbl_op_ds_w_rayleigh.to_netcdf(fp_w)

# perturbation sims
perturbation_op_ds_fp = intermidiate_data_storage_path + 'profile_v{}/lblrtm_op_ds_tonga.nc'.format(profile_id)
if os.path.exists(perturbation_op_ds_fp):
    print('Reusing local copy of the previous Tonga LBLRTM calculations')
    lbl_op_ds_tonga = xarray.open_dataset(perturbation_op_ds_fp)
else:
    print('\nPreparing LBLRTM run for Tonga perturbation\n')
    lbl_op_ds_tonga = run_lblrtm_over_spectral_range(SPECTRAL_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_tonga, cross_sections, lblrtm_setup_vo, False)
    make_dir_for_the_full_file_path(perturbation_op_ds_fp)
    lbl_op_ds_tonga.to_netcdf(perturbation_op_ds_fp)
#%% MIX OP: derive the mixture optical properties + do the physical validation and fixes
print('Checking in: op_ds_wo_rayleigh')
checkin_and_fix(lbl_op_ds_wo_rayleigh)
print('Checking in: op_ds_w_rayleigh')
checkin_and_fix(lbl_op_ds_w_rayleigh)
print('Checking in: lbl_op_ds_tonga')
checkin_and_fix(lbl_op_ds_tonga)

# TODO: Would be best to parametrize the Rayleigh optical depth instead of running LBLRTM twice
rayleigh_op_ds = derive_rayleigh_optical_properties(lbl_op_ds_w_rayleigh, lbl_op_ds_wo_rayleigh)
print('Checking in: rayleigh_op_ds')
checkin_and_fix(rayleigh_op_ds)

# get the mixtures of gas absorption & Rayleigh scattering
mixed_op_ds_ref = mix_optical_properties([lbl_op_ds_wo_rayleigh, rayleigh_op_ds], externally=True)
mixed_op_ds_tonga = mix_optical_properties([lbl_op_ds_tonga, rayleigh_op_ds], externally=True)
# this should be excessive
print('Checking in: mixed_op_ds_ref')
checkin_and_fix(mixed_op_ds_ref)
print('Checking in: mixed_op_ds_tonga')
checkin_and_fix(mixed_op_ds_tonga)

#%% run disort
disort_setup_vo = DisortSetup()
setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)
setup_surface_albedo(disort_setup_vo)
disort_setup_vo.albedo = 0.06  # setup ocean albedo
disort_setup_vo.ONLYFL = True

wn_grid_step = mixed_op_ds_ref.wavenumber[1] - mixed_op_ds_ref.wavenumber[0]
disort_setup_vo.wn_grid_step = wn_grid_step.item()  # width of wn range in cm-1. From LBLRTM TODO: think how to immplement this fail safe

disort_ds_fp = intermidiate_data_storage_path + 'disort_output_ref.nc'
if os.path.exists(disort_ds_fp):
    print('Reusing the local copy of the DISORT Ref')
    disort_output_ds_ref = xr.open_dataset(disort_ds_fp)
else:
    disort_output_ds_ref = run_disort_spectral(mixed_op_ds_ref, atm_stag_ds, disort_setup_vo)
    make_dir_for_the_full_file_path(disort_ds_fp)
    disort_output_ds_ref.to_netcdf(disort_ds_fp)

# perturbation sims
disort_ds_fp = intermidiate_data_storage_path + '{}/disort_output_tonga.nc'.format(scenario_label)
if os.path.exists(disort_ds_fp):
    print('Reusing the local copy of the DISORT Tonga')
    disort_output_ds_tonga = xr.open_dataset(disort_ds_fp)
else:
    disort_output_ds_tonga = run_disort_spectral(mixed_op_ds_tonga, atm_stag_ds_tonga, disort_setup_vo)
    make_dir_for_the_full_file_path(disort_ds_fp)
    disort_output_ds_tonga.to_netcdf(disort_ds_fp)

# %% run disort: only DAYTIME part of the DIURNAL CYCLE
do_diurnal_cycle = True
if do_diurnal_cycle:
    diurnal_cycle_hours = np.arange(0, 24, 0.5)

    disort_ds_fp = intermidiate_data_storage_path + '/diurnal/disort_output_ref.nc'
    if os.path.exists(disort_ds_fp):
        print('Reusing the local copy of the DISORT Ref')
        disort_output_ds_dc_ref = xr.open_dataset(disort_ds_fp)
    else:
        diurnal_cycle_szas = []
        diurnal_cycle_dates = []
        dss = []
        for hour in diurnal_cycle_hours:
            print('Running DISORT diurnal cycle hour {} out of {}'.format(hour, diurnal_cycle_hours[-1]))
            diurnal_cycle_date = date + dt.timedelta(hours=hour)
            setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], diurnal_cycle_date)
            print('SZA is {}'.format(disort_setup_vo.zenith_angle_degree))

            if disort_setup_vo.zenith_angle_degree >= 90:
                print('Sun is below the horizon, skipping')
                continue

            diurnal_cycle_dates += [diurnal_cycle_date, ]
            diurnal_cycle_szas += [disort_setup_vo.zenith_angle_degree, ]

            # sw_op_ds = mixed_op_ds_ref.where((mixed_op_ds_ref.wavenumber>=RRTM_SW_WN_RANGE[0]) & (mixed_op_ds_ref.wavenumber<=RRTM_SW_WN_RANGE[1]), drop=True)
            ds = run_disort_spectral(mixed_op_ds_ref, atm_stag_ds, disort_setup_vo)
            dss += [ds]

        disort_output_ds_dc_ref = xr.concat(dss, dim='time')
        disort_output_ds_dc_ref['time'] = diurnal_cycle_dates
        disort_output_ds_dc_ref['sza'] = diurnal_cycle_szas
        make_dir_for_the_full_file_path(disort_ds_fp)
        disort_output_ds_dc_ref.to_netcdf(disort_ds_fp)

    # perturbation sims
    disort_ds_fp = intermidiate_data_storage_path + '/diurnal/{}/disort_output_tonga.nc'.format(scenario_label)
    if os.path.exists(disort_ds_fp):
        print('Reusing the local copy of the DISORT Tonga')
        disort_output_ds_dc_tonga = xr.open_dataset(disort_ds_fp)
    else:
        diurnal_cycle_szas = []
        diurnal_cycle_dates = []
        dss = []
        for hour in diurnal_cycle_hours:
            print('Running DISORT diurnal cycle hour {} out of {}'.format(hour, diurnal_cycle_hours[-1]))
            diurnal_cycle_date = date + dt.timedelta(hours=hour)
            setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], diurnal_cycle_date)
            print('SZA is {}'.format(disort_setup_vo.zenith_angle_degree))

            if disort_setup_vo.zenith_angle_degree >= 90:
                print('Sun is below the horizon, skipping')
                continue

            diurnal_cycle_dates += [diurnal_cycle_date, ]
            diurnal_cycle_szas += [disort_setup_vo.zenith_angle_degree, ]

            # sw_op_ds = mixed_op_ds_tonga.where((mixed_op_ds_tonga.wavenumber >= RRTM_SW_WN_RANGE[0]) & (mixed_op_ds_tonga.wavenumber <= RRTM_SW_WN_RANGE[1]), drop=True)
            ds = run_disort_spectral(mixed_op_ds_tonga, atm_stag_ds_tonga, disort_setup_vo)
            dss += [ds]

        disort_output_ds_dc_tonga = xr.concat(dss, dim='time')
        disort_output_ds_dc_tonga['time'] = diurnal_cycle_dates
        disort_output_ds_dc_tonga['sza'] = diurnal_cycle_szas
        make_dir_for_the_full_file_path(disort_ds_fp)
        disort_output_ds_dc_tonga.to_netcdf(disort_ds_fp)
#%% PREP for plotting
op_keys = ['od', 'ssa', 'g']

ds = disort_output_ds_ref
disort_keys = []
for key in ds.variables:
    if ds[key].ndim == 2:
        disort_keys += [key, ]
#%% DEBUGGING
debug = False
if debug:
    plt.ion()
    ds = disort_output_ds_tonga
    ds = disort_output_ds_ref
    plt.clf()
    ds.isel(level=-1).diffuse_flux_down.plot(xscale='log')

    toa_ds = ds.isel(level=-1)
    neg_ds = toa_ds.where(toa_ds.diffuse_flux_down<0, drop=True)
    print('Wavenumbers with negative dfd {}'.format(neg_ds.wavenumber.data))
    neg_ds = toa_ds.where(toa_ds.diffuse_flux_down<-10**-3, drop=True)
    print('Wavenumbers with negative dfd {}'.format(neg_ds.wavenumber.data))
    neg_ds = toa_ds.where(toa_ds.diffuse_flux_down<-2, drop=True)
    print('Wavenumbers with negative dfd {}'.format(neg_ds.wavenumber.data))

    wn = 910
    wn = 1040
    # plot input profile
    op_ds = mixed_op_ds_ref
    # op_ds = mixed_op_ds_ref_lw_abs_sw_sca
    axes = plot_spectral_profiles(op_ds.sel(wavenumber=wn), op_keys)
    axes[1].set_xscale('linear')
    plt.suptitle('Ref (SW sca, LW abs)')
    ax = axes.flatten()[-1]
    plt.cla()
    atm_stag_ds.t.plot(y='level', yscale='log')
    ax.invert_yaxis()
    save_figure(pics_folder, 'input_profile_ref_{}'.format(wn))

    # disport output profile
    axes = plot_spectral_profiles(ds.sel(wavenumber=wn), disort_keys, xscale='linear')
    plt.suptitle('DISORT: Ref, LW Abs')
    save_figure(pics_folder, 'disort_profile_ref_abs_lw_{}'.format(wn))


#%% let have a look at the input profiles
fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)  #, figsize=(2*JGR_page_width_inches(), JGR_page_width_inches()))
ax = axes[0]
gases_ds_ref.sel(species=Gas.H2O.value).const.plot(ax=ax, y='level', marker='o', label='ref')
gases_ds_tonga.sel(species=Gas.H2O.value).const.plot(ax=ax, y='level', marker='*', label='Tonga')
ax.legend()
ax.invert_yaxis()
ax.set_yscale('log')
ax.set_xscale('log')
# plt.xlabel('Volume mixing ratio (ppmv)')
ax = axes[1]
atm_stag_ds.t.plot(ax=ax, y='level', marker='o', label='ref')
atm_stag_ds_tonga.t.plot(ax=ax, y='level', marker='*', label='Tonga')
ax.legend()
ax.invert_yaxis()
ax.set_yscale('log')
# ax.set_xscale('log')
save_figure(pics_folder, 'input profiles')

axes[0].set_ylim([100, 1])
axes[1].set_ylim([100, 1])
axes[0].set_xlim([10**0, 10**2])
# axes[1].set_xlim([210, 240])
save_figure(pics_folder, 'input profiles, h2o, zoomed')

#%% PLOT input profile zoomed in
# gases_ds_ref['z'] = atm_stag_ds.z.data
# gases_ds_tonga['z'] = atm_stag_ds.z
plt.figure()
gases_ds_ref.sel(species=Gas.H2O.value).const.plot(y='level', label='ref', marker='o', yincrease=False, yscale='log')
gases_ds_tonga.sel(species=Gas.H2O.value).const.plot(y='level', marker='*', label='Tonga')
plt.ylim([100,1])
plt.xlim([0,50])
save_figure(pics_folder, 'input profiles, h2o, zoomed')
#%% plot OPs
op_keys = ['od', 'ssa', 'g']
axes = plot_spectral_profiles(mixed_op_ds_ref, op_keys)
plt.suptitle('Ref')
ax = axes.flatten()[-1]
rayleigh_op_ds.sum(dim='level').od.plot(ax = ax, label='Rayleigh')
mixed_op_ds_ref.sum(dim='level').od.plot(ax = ax, label='Mixed', xscale='log', yscale='log')
plt.legend()
plt.title('Column OD')
save_figure(pics_folder, 'op_mixed_ref')

plot_spectral_profiles(mixed_op_ds_tonga, op_keys)
plt.suptitle('Tonga')
save_figure(pics_folder, 'op_mixed_tonga')
ds = mixed_op_ds_tonga-mixed_op_ds_ref
axes = plot_spectral_profiles(ds, op_keys)
plt.suptitle('Tonga-Ref')
ds.sum(dim='level').od.plot(ax = axes.flatten()[-1])
plt.title('Column OD')
plt.xscale('log')
plt.yscale('log')
save_figure(pics_folder, 'op_mixed_pmc')

axes = plot_spectral_profiles(lbl_op_ds_wo_rayleigh, op_keys)
plt.suptitle('LBLRTM Ref, no Rayleigh')
ax = axes.flatten()[-1]
plt.cla()
lbl_op_ds_wo_rayleigh.sum(dim='level').od.plot(ax = ax)
plt.title('Column OD')
ax.set_xscale('log')
ax.set_yscale('log')
save_figure(pics_folder, 'op_gases_ref_wo_rayleigh')

axes = plot_spectral_profiles(rayleigh_op_ds, op_keys)
plt.suptitle('Rayleigh')
ax = axes.flatten()[-1]
plt.cla()
rayleigh_op_ds.sum(dim='level').od.plot(ax = ax)
plt.title('Rayleigh column OD')
ax.set_xscale('log')
ax.set_yscale('log')
save_figure(pics_folder, 'op_rayleigh')

# do the profile at LW wavelength, 10 um
wn = 10**4/10
axes = plot_spectral_profiles(mixed_op_ds_ref.sel(wavenumber=wn), op_keys)
plt.suptitle('Ref')
ax = axes[-1]
ax.set_xscale('linear')
save_figure(pics_folder, 'op_mixed_ref_{}'.format(wn))

#%% plot the DISORT output spectral
axes = plot_spectral_profiles(disort_output_ds_ref, disort_keys)
plt.suptitle('DISORT: Ref')
save_figure(pics_folder, 'profiles_disort_ref_spectral')

plot_spectral_profiles(disort_output_ds_tonga, disort_keys)
plt.suptitle('DISORT: Tonga')
save_figure(pics_folder, 'profiles_disort_tonga_spectral')

plot_spectral_profiles(disort_output_ds_tonga - disort_output_ds_ref, disort_keys)
plt.suptitle('DISORT: Tonga-Ref')
save_figure(pics_folder, 'profiles_disort_pmc_spectral')

#%% plot the DISORT output BROADBAND
model_label = 'DISORT'
wn_range_labels = ['SW', 'LW', 'NET (SW+LW)']
wn_ranges = [slice(RRTM_LW_WN_RANGE[1], None), slice(0, RRTM_LW_WN_RANGE[1]), slice(None, None), ]
for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    axes1, axes2, axes3 = plot_ref_perturbed_pmc(wn_range, wn_range_label, 'Tonga', disort_output_ds_ref, disort_output_ds_tonga, disort_keys, model_label, pics_folder)

    if profile_id == 3 and wn_range_label == 'SW':  # reduce ticks frequency for a specific case
        print('profile')
        axes3[0, 0].set_xlim(left=-0.175)
        axes3[1, 1].set_xlim(left=-0.175)
        axes3[2, 0].set_xlim(left=-0.175)
        save_figure(pics_folder, 'profiles_disort_pmc_{}'.format(wn_range_label))

#%% plot the DISORT output BROADBAND & DIURNAL (ONLY SW)
for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    axes1, axes2, axes3 = plot_ref_perturbed_pmc(wn_range, wn_range_label, 'Tonga', disort_output_ds_dc_ref, disort_output_ds_dc_tonga, disort_keys, model_label, pics_folder+'diurnal/')

#%% mean DC
for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    axes1, axes2, axes3 = plot_ref_perturbed_pmc(wn_range, wn_range_label, 'Tonga', disort_output_ds_dc_ref.sum(dim='time') / len(diurnal_cycle_hours), disort_output_ds_dc_tonga.sum(dim='time') / len(diurnal_cycle_hours), disort_keys, model_label, pics_folder+'diurnal_mean/')

#%% Estimate diurnal averaging via SZA. The weight for SW forcing should be 0.35
import pvlib
diurnal_cycle_hours = np.arange(0, 24, 0.5)
diurnal_cycle_szas = []
for hour in diurnal_cycle_hours:
    diurnal_cycle_date = date + dt.timedelta(hours=hour)
    solpos = pvlib.solarposition.get_solarposition(diurnal_cycle_date, lats[0], lons[0], 0)
    diurnal_cycle_szas += [solpos['zenith'].iloc[0], ]

df = pd.DataFrame({'sza':diurnal_cycle_szas, 'hour':diurnal_cycle_hours})
df['cos(sza)'] = np.cos(np.deg2rad(diurnal_cycle_szas))
df['solar_weight'] = df['cos(sza)']
df['solar_weight'][df['solar_weight']<0] = 0

df.plot(x='hour', y='sza')
ax = df.plot(x='hour', y='cos(sza)')
df.plot(ax=ax, x='hour', y='solar_weight')
plt.title('Mean cos(sza) above horizon is {:.2f}'.format(df['solar_weight'].mean()))