import os
import os.path

import numpy as np
import xarray
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt

from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.optics_utils import derive_rayleigh_optical_properties, mix_optical_properties
from climpy.utils.plotting_utils import save_figure, JGR_page_width_inches
from lblrtm_utils import write_settings_to_tape, Gas, run_lblrtm, read_od_output, LblrtmSetup, run_lblrtm_over_spectral_range
from disort_utils import run_disort, DisortSetup, run_disort_spectral, setup_viewing_geomtry, \
    prep_chanceetal_sun_spectral_irradiance, setup_surface_albedo, RRTM_SW_WN_RANGE, RRTM_SW_LW_WN_RANGE, RRTM_LW_WN_RANGE

intermidiate_data_storage_path = '/Users/osipovs/Data/Tonga/'
pics_folder = get_pictures_root_folder()+'Tonga/'
#%% experimental setup
lats = np.array([-20.55,])  # location & date to process
lons = np.array([175.385,])
date = dt.datetime(1991, 1, 2)
#%% setup atmospheric meteorological profile


def get_atmospheric_profile():
    fp = '{}sfc/{}'.format('/work/mm0062/b302074/Data/ECMWF/EraInterim/netcdf/global/F128/', 'ECMWF_sfc_19910102_19910102.nc')
    fp = os.path.expanduser('~') + '/Temp/ECMWF_sfc_19910102_19910102.nc'  # temp local
    sfc_ds = xr.open_dataset(fp)
    sfc_ds = sfc_ds.rename_vars({'z': 'z_sfc'})
    fp = '{}pl/{}'.format('/work/mm0062/b302074/Data/ECMWF/EraInterim/netcdf/global/F128/', 'ECMWF_pl_19910102_19910102.nc')
    fp = os.path.expanduser('~') + '/Temp/ECMWF_pl_19910102_19910102.nc'  # temp local
    profile_ds = xr.open_dataset(fp)

    # add pressure variable
    profile_ds['p'] = (('level', ), profile_ds.level.data)  # keep in 1D although general approach is N-D

    # reverse the z direction to have indexing start at the surface
    profile_ds = profile_ds.sel(level=slice(None, None, -1))

    # merge surface and profile datasets
    ds = xr.merge([sfc_ds, profile_ds])
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # correct few things
    ds.sp[:] /= 10**2
    ds.sp.attrs['units'] = 'hPa'

    # convert geopotential to height in meters
    # TODO: it is probably faster to do it after location sampling or in output parser
    g = 9.8  # m/sec**2
    ds.z_sfc[:] /= g
    ds.z_sfc.attrs['units']='m'
    ds.z_sfc.attrs['long_name'] = 'Height'
    ds.z_sfc.attrs['standard_name'] = 'height'

    ds.z[:] /= g
    ds.z.attrs['units'] = 'm'
    ds.z.attrs['long_name'] = 'Height'
    ds.z.attrs['standard_name'] = 'height'

    return ds


atm_stag_ds = get_atmospheric_profile()  # stag indicates staggered profile
atm_stag_ds = atm_stag_ds.sel(lat=lats, lon=lons, time=date, method='nearest')  # TODO: date selection should be controlled by tollerance
atm_stag_ds['p'] = (('level', 'lat', 'lon'), atm_stag_ds['p'].data[:, np.newaxis, np.newaxis])  # keep in 1D although general approach is N-D

# atm_stag_ds.z[0, 0, 0] = 0  # TODO: temp test is z has to start from zero
atm_stag_ds = atm_stag_ds.isel(level=range(len(atm_stag_ds.level) - 2))
#%% setup atmospheric chemical composition (gases)


def get_atmospheric_gases_composition():
    fp = '/work/mm0062/b302074/Data/NASA/GMI/gmiClimatology.nc'  # GMI climatology
    fp = os.path.expanduser('~') + '/Data/NASA/GMI/gmiClimatology.nc'  # GMI climatology
    ds = xr.open_dataset(fp)
    # fix the metadata
    ds = ds.set_coords({'lat', 'lon', 'level'})
    ds = ds.swap_dims({'latitude_dim': 'lat', 'longitude_dim': 'lon', 'eta_dim': 'level'})
    ds = ds.rename({'species_dim': 'species'})

    species = []  # derive labels from netcdf
    for index in range(ds.const_labels.shape[1]):
        label = ''.join([r.item().decode().strip() for r in ds.const_labels[:,index]])
        species += [label, ]

    ds['species'] = (('species', ), species)

    # SO2 is missing, setup dummy profile.
    so2_ds = ds.sel(species=Gas.O2.value)  # Use O2 to make a single gas copy as a template for SO2
    so2_ds.const[:]=0
    so2_ds['species'] = (('species',), ['SO2',])

    # CO2 is missing, setup 388.5 (as of june 2012).
    co2_ds = ds.sel(species=Gas.O2.value)  # Use O2 to make a single gas copy as a template for SO2
    co2_ds.const[:] = 388.5  # units A
    co2_ds['species'] = (('species',), ['CO2', ])

    ds = xr.concat([ds, so2_ds, co2_ds], 'species')
    ds['time'] = np.arange(12)  # zero based indexing

    ds = ds.drop(labels=('const_labels',))

    # add units variable according to LBLRTM
    ds['units'] = (('species',), ['A',]*ds.species.size)
    ds.units.attrs['long_name'] = 'units according to LBLRTM'

    return ds


def rel_humidity_to_mass_concentration(atm_stag_ds):
    # work with concentration instead of relative humidity
    from metpy.units import units
    from metpy.calc import mixing_ratio_from_relative_humidity, density

    # kg/kg
    h2o_ppm = mixing_ratio_from_relative_humidity(atm_stag_ds.p.values * units.hPa, atm_stag_ds.t.values * units.degK, atm_stag_ds.r.values/100)  # .to('g/kg')
    # kg/m^3
    air_density = density(atm_stag_ds.p.values * units.hPa, atm_stag_ds.t.values * units.degK, 0 * units('g/kg'))  # otherwise provide water mixing ration and get wet air density
    h2o_mass_concentration = h2o_ppm * air_density  # kg / m^3
    return h2o_mass_concentration  # kg / m^3


#%% setup H2O from atm grid (relative humidity)
gases_ds = get_atmospheric_gases_composition()
gases_ds = gases_ds.sel(lat=lats, lon=lons, time=date.month-1, method='nearest')  # do the selection
# interpolate gases on the LBLRTM vertical grid
gases_ds = gases_ds.interp(level=atm_stag_ds.level.data, kwargs={"fill_value": "extrapolate"})  # TODO: extrapolating blindly is bad
# Add H2O, already on the MERRA2 grid
h2o_ds = gases_ds.sel(species=Gas.O2.value).copy(deep=True)  # Use O2 to make a single gas copy as a template for SO2

# relative humidity
h2o_ds.const[:] = atm_stag_ds.r[:].data  # units H # TODO: MERRA2 has negative relative humidity values, fix them
h2o_ds.const.attrs['units'] = 'relative humidity (%)'
h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
h2o_ds['units'] = (('species',), ['H',])  # relative humidity
# alternative H2O units
# h2o_ds.const[:] = rel_humidity_to_mass_concentration(atm_stag_ds) * 10**3  # A-ppmv, D-mass density (gm m-3)
# h2o_ds.const.attrs['units'] = 'mass density (gm m-3)'
# h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
# h2o_ds['units'] = (('species',), ['D',])  # mass density (gm m-3)

gases_ds = xr.concat([gases_ds, h2o_ds], 'species')

gases_ds_ref = gases_ds
#%% adjust H2O for reference & perturbed cases

# set RH to 50% in ref and 100% in Tonga sims
h2o_ds = gases_ds_ref.sel(species=Gas.H2O.value)
h2o_ds.sel(level=slice(100,5))['const'][:] = 50

gases_ds_tonga = gases_ds_ref.copy(deep=True)
h2o_ds = gases_ds_tonga.sel(species=Gas.H2O.value)
h2o_ds.sel(level=slice(100,5))['const'][:] = 100
# levels = h2o_ds['const'].level
# h2o_ds['const'].loc[dict(level=levels[levels<100])] *= 1.1  # boost water vapor above 100 hpa by 10%

#%% estimate LNFL & LBLRTM spectral requirements
'''
LNLFL: The TAPE3 file should include lines from at least 25 cm-1 on either end of the calculation region.
'''
print('LNFL TAPE5 range for SW should be: {} {}'.format(RRTM_SW_LW_WN_RANGE[0] - 25, RRTM_SW_LW_WN_RANGE[1] + 25))
#%% derive gases absorption OD & optical properties
if os.path.exists(intermidiate_data_storage_path + 'op_ds_w_rayleigh.nc'):
    print('Reusing local copy of the previos LBLRTM calculations')
    op_ds_w_rayleigh = xarray.open_dataset(intermidiate_data_storage_path + 'op_ds_w_rayleigh.nc')
    op_ds_wo_rayleigh = xarray.open_dataset(intermidiate_data_storage_path + 'op_ds_wo_rayleigh.nc')
    op_ds_tonga = xarray.open_dataset(intermidiate_data_storage_path + 'op_ds_tonga.nc')
else:
    print('Runing LBLRTM clean')
    # cross_sections = gases_ds.sel(species=[Gas.NO2.value, Gas.SO2.value])  # indicate the gases, for which the xsections should be accounted for
    cross_sections = gases_ds.sel(species=[Gas.NO2.value, ])  # indicate the gases, for which the xsections should be accounted for

    lblrtm_scratch_fp = '/Users/osipovs/Temp/'  # local
    lblrtm_scratch_fp = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/run_tonga/'  # remote

    # calculate optical depth only (gas absorption + rayleigh). Remember that LBLRTM represent rho layer and optical properties of these layers
    print('\nPreparing LBLRTM run with Rayleigh scattering\n')
    op_ds_w_rayleigh = run_lblrtm_over_spectral_range(RRTM_SW_LW_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, True)
    op_ds_w_rayleigh.to_netcdf(intermidiate_data_storage_path + 'op_ds_w_rayleigh.nc')
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    op_ds_wo_rayleigh = run_lblrtm_over_spectral_range(RRTM_SW_LW_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, False)
    op_ds_wo_rayleigh.to_netcdf(intermidiate_data_storage_path + 'op_ds_wo_rayleigh.nc')
    print('\nPreparing LBLRTM run for Tonga perturbation\n')
    op_ds_tonga = run_lblrtm_over_spectral_range(RRTM_SW_LW_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_tonga, cross_sections, False)
    op_ds_tonga.to_netcdf(intermidiate_data_storage_path + 'op_ds_tonga.nc')

#%% derive the mixture optical properties + do the physical validation and fixes
def checkin_and_fix(ds):
    if ds.od.min()<0:
        if np.abs(ds.od.min()) < 10**-6:
            print('Min')
            print(ds.min())
            print('Max')
            print(ds.max())

            count = (ds.od < 0).sum()
            print('Zeroing out {} negative OD values'.format(count.item()))

            ds.od.where(ds.od<0)[:] = 0
            ds['od'] = ds.od.where(ds.od>0).fillna(0)
        else:
            raise Exception('Dataset contains negative values beyound the tollerance. Check what is wrong.')


# derive Rayleigh first and then check in the LBLRTM output
print('Checking in: op_ds_wo_rayleigh')
checkin_and_fix(op_ds_wo_rayleigh)
print('Checking in: op_ds_w_rayleigh')
checkin_and_fix(op_ds_w_rayleigh)

# Rayleigh scattering is following the Nicolet 1984 https://doi.org/10.1016/0032-0633(84)90089-8
# https://doi.org/10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2
# TODO: Would be best to parametrize the Rayleigh optical depth instead of running LBLRTM twice
rayleigh_op_ds = derive_rayleigh_optical_properties(op_ds_w_rayleigh, op_ds_wo_rayleigh)
print('Checking in: rayleigh_op_ds')
checkin_and_fix(rayleigh_op_ds)
# get the mixtures of gas absorption & Rayleigh scattering
mixed_op_ds_ref = mix_optical_properties([op_ds_wo_rayleigh, rayleigh_op_ds], externally=True)
mixed_op_ds_tonga = mix_optical_properties([op_ds_tonga, rayleigh_op_ds], externally=True)
# this should be excessive
print('Checking in: mixed_op_ds_ref')
checkin_and_fix(mixed_op_ds_ref)
print('Checking in: mixed_op_ds_tonga')
checkin_and_fix(mixed_op_ds_tonga)

#%% run disort
if os.path.exists(intermidiate_data_storage_path + 'disort_output_ref.nc'):
    disort_output_ds_ref = xr.open_dataset(intermidiate_data_storage_path + 'disort_output_ref.nc')
    disort_output_ds_tonga = xr.open_dataset(intermidiate_data_storage_path + 'disort_output_tonga.nc')
else:
    disort_setup_vo = DisortSetup()
    setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)
    setup_surface_albedo(disort_setup_vo)
    disort_setup_vo.albedo = 0.06  # setup ocean albedo
    # disort_setup_vo.NMOM = 64  # lets see if it fixes negative fluxes

    wn_grid_step = mixed_op_ds_ref.wavenumber[1] - mixed_op_ds_ref.wavenumber[0]
    disort_setup_vo.wn_grid_step = wn_grid_step.item()  # width of wn range in cm-1. From LBLRTM TODO: think how to immplement this fail safe

    op_ds = mixed_op_ds_ref
    disort_output_ds_ref = run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo)
    disort_output_ds_ref.to_netcdf(intermidiate_data_storage_path + 'disort_output_ref.nc')

    op_ds = mixed_op_ds_tonga
    disort_output_ds_tonga = run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo)
    disort_output_ds_tonga.to_netcdf(intermidiate_data_storage_path + 'disort_output_tonga.nc')

#%% pp: derive down minus up diag
ds = disort_output_ds_ref
ds['down_minus_up_flux'] = ds.direct_flux_down+ds.diffuse_flux_down-ds.diffuse_flux_up
ds = disort_output_ds_tonga
ds['down_minus_up_flux'] = ds.direct_flux_down+ds.diffuse_flux_down-ds.diffuse_flux_up

#%%
# print('MASKING OUT unphysical data')
# disort_output_ds_ref['diffuse_flux_down'] = disort_output_ds_ref.diffuse_flux_down.where(disort_output_ds_ref.diffuse_flux_down > -10**-2, 0)
# plt.figure()
# plt.clf()
# asd.plot()

#%% let have a look at the input profiles
plt.clf()
gases_ds_ref.sel(species=Gas.H2O.value).const.plot(y='level', marker='o', label='ref')
gases_ds_tonga.sel(species=Gas.H2O.value).const.plot(y='level', marker='*', label='Tonga')
plt.legend()
plt.gca().invert_yaxis()
plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.xlabel('Relative Humidity (%)')
save_figure(pics_folder, 'input profiles')


#%% plot the profiles
def plot_spectral_profiles(ds, keys):
    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(2*JGR_page_width_inches(), JGR_page_width_inches()))
    indexer = 0
    for var_key in keys:
        var_ds = ds[var_key]
        ax = axes.flatten()[indexer]
        var_ds.plot(ax=ax, y='level')

        ax.invert_yaxis()
        ax.set_yscale('log')
        if var_ds.ndim > 1:
            ax.set_xscale('log')

        print('{}: Min is {}, Max is {}'.format(var_key, var_ds.min().item(), var_ds.max().item()))
        indexer += 1

    return axes


#%% plot OPs
op_keys = ['od', 'ssa', 'g']
axes = plot_spectral_profiles(mixed_op_ds_ref, op_keys)
plt.suptitle('Ref')
ax = axes.flatten()[-1]
rayleigh_op_ds.sum(dim='level').od.plot(ax = ax, label='Rayleigh')
mixed_op_ds_ref.sum(dim='level').od.plot(ax = ax, label='Mixed')
plt.legend()
plt.title('Column OD')
ax.set_xscale('log')
ax.set_yscale('log')
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

axes = plot_spectral_profiles(op_ds_wo_rayleigh, op_keys)
plt.suptitle('LBLRTM Ref, no Rayleigh')
ax = axes.flatten()[-1]
op_ds_wo_rayleigh.sum(dim='level').od.plot(ax = ax)
plt.title('Column OD')
ax.set_xscale('log')
ax.set_yscale('log')
save_figure(pics_folder, 'op_gases_ref_wo_rayleigh')

axes = plot_spectral_profiles(rayleigh_op_ds, op_keys)
plt.suptitle('Rayleigh')
ax = axes.flatten()[-1]
rayleigh_op_ds.sum(dim='level').od.plot(ax = ax)
plt.title('Rayleigh column OD')
ax.set_xscale('log')
ax.set_yscale('log')
save_figure(pics_folder, 'op_rayleigh')
#%%
# plot_spectral_profiles(op_ds_tonga-op_ds_wo_rayleigh, op_keys)
# plt.suptitle('Tonga-Ref, gases only')
# save_figure(pics_folder, 'op_pmc')

#%% plot the DISORT output
ds = disort_output_ds_ref
disort_keys = []
for key in ds.variables:
    if ds[key].ndim == 2:
        disort_keys += [key, ]

axes = plot_spectral_profiles(disort_output_ds_ref, disort_keys)
plt.title('DISORT: Ref')
save_figure(pics_folder, 'profiles_disort_ref_spectral')
plot_spectral_profiles(disort_output_ds_tonga, disort_keys)
plt.title('DISORT: Tonga')
save_figure(pics_folder, 'profiles_disort_tonga_spectral')
plot_spectral_profiles(disort_output_ds_tonga-disort_output_ds_ref, disort_keys)
plt.title('DISORT: Tonga-Ref')
save_figure(pics_folder, 'profiles_disort_pmc_spectral')

#%% debug
# count = (ref_disort_output_ds.sel(level=3).diffuse_flux_down != 0).sum()
# ref_disort_output_ds.sel(level=3).diffuse_flux_down.plot(marker='*')
# plt.xscale('log')

# NET
model_label = 'DISORT'
wn_range_label = 'net (sw+lw)'
plot_spectral_profiles(disort_output_ds_ref.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nRef'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profile_disort_ref')
plot_spectral_profiles(disort_output_ds_tonga.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nTonga'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profile_disort_tonga')
ds = disort_output_ds_tonga - disort_output_ds_ref
plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nTonga-Ref'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profile_disort_pmc')

# SW or LW
model_label = 'DISORT'
wn_range_label = 'net (sw+lw)'
wn_range = slice(None, None)  # SW
wn_range_label = 'sw'
wn_range = slice(RRTM_LW_WN_RANGE[1], None)  # SW
wn_range_label = 'lw'
wn_range = slice(0, RRTM_LW_WN_RANGE[1])

ds = disort_output_ds_ref.sel(wavenumber=wn_range)
plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nRef'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profiles_disort_ref_{}'.format(wn_range_label))

ds = disort_output_ds_tonga.sel(wavenumber=wn_range)
plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nTonga'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profiles_disort_tonga_{}'.format(wn_range_label))

ds = disort_output_ds_tonga - disort_output_ds_ref
ds = ds.sel(wavenumber=wn_range)
plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys)
plt.suptitle('{}. {}\nTonga-Ref'.format(model_label, wn_range_label))
save_figure(pics_folder, 'profiles_disort_pmc_{}'.format(wn_range_label))


#%%
# search for the outliers
ds = disort_output_ds_tonga
asd = ds.sel(wavenumber=wn_range).diffuse_flux_down
asd = asd.where(asd>0).fillna(0)
plt.clf()
asd.integrate('wavenumber').plot(marker='o')

ds = disort_output_ds_ref
plt.clf()
ds.sel(level=3).diffuse_flux_down.plot(marker='o')

plt.clf()
ds.sel(level=3).sel(wavenumber=slice(300,500)).diffuse_flux_down.plot(marker='o')
ds.sel(level=3).sel(wavenumber=slice(300,500)).direct_flux_down.plot(marker='o')
ds.sel(level=3).sel(wavenumber=slice(300,500)).diffuse_flux_up.plot(marker='o')
#%%
def plot_rf_profile(ax, ds):
    ds = ds.integrate('wavenumber')  # I get NET=SW+LW
    dmu_ds = ds.direct_flux_down+ds.diffuse_flux_down-ds.diffuse_flux_up  # down minus up
    dmu_ds.name = 'NET (SW+LW) Down minus Up flux'
    # fig, axes = plt.subplots(constrained_layout=True, figsize=(2*JGR_page_width_inches(), JGR_page_width_inches()))
    dmu_ds.plot(ax=ax, y='level')
    ax.invert_yaxis()
    ax.set_yscale('log')


fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(2*JGR_page_width_inches(), JGR_page_width_inches()))
# plt.clf()
plot_rf_profile(axes.flatten()[0], disort_output_ds_ref)
plot_rf_profile(axes.flatten()[1], disort_output_ds_ref.sel(wavenumber=))
plot_rf_profile(axes.flatten()[2], disort_output_ds_ref.sel(wavenumber=))
plt.title('Ref')
save_figure(pics_folder, 'profile_rf_tonga')
plot_rf_profile(disort_output_ds_tonga - disort_output_ds_ref)
save_figure(pics_folder, 'profile_rf_pmc')

#%%
# toa_ds = ref_disort_output_ds.sel(level=3).integrate('wavenumber')

#%%

#%% sun irradiance
diff_ds = disort_output_ds_tonga.direct_flux_down - disort_output_ds_ref.direct_flux_down
plt.clf()
# diff_ds.sortby('wavelength').plot()
diff_ds.plot()

#%%
plt.clf()
op_diff_ds = op_ds_tonga - op_ds_w_rayleigh
op_diff_ds.od.plot()
#%%
# gases_diff_ds = gases_ds_tonga-gases_ds_ref
h2o_diff_ds = gases_ds_tonga.sel(species=Gas.H2O.value).drop('units') - gases_ds_ref.sel(species=Gas.H2O.value).drop('units')
plt.clf()
h2o_diff_ds.const.plot(marker='o')
# h2o_ds.const.plot()