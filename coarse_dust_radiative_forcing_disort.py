import os
import os.path
import climpy.utils.mie_utils as mie
import numpy as np
import xarray
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import scipy as sp
from climpy.utils.atmos_chem_utils import get_atmospheric_gases_composition
from climpy.utils.atmos_utils import get_atmospheric_profile
from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.ncar_obs_utils import get_probed_size_distribution_in_riyadh
from climpy.utils.optics_utils import derive_rayleigh_optical_properties, mix_optical_properties
from climpy.utils.plotting_utils import save_figure, JGR_page_width_inches
from climpy.utils.lblrtm_utils import Gas, run_lblrtm_over_spectral_range, LblrtmSetup
from climpy.utils.disort_utils import DisortSetup, run_disort_spectral, setup_viewing_geomtry, \
    setup_surface_albedo, RRTM_SW_LW_WN_RANGE, RRTM_LW_WN_RANGE, checkin_and_fix
from climpy.utils.refractive_index_utils import get_dust_ri
from disort_plotting_utils import plot_spectral_profiles, plot_ref_perturbed_pmc

intermediate_data_storage_path = '/Users/osipovs/Data/DustRF/'
pics_folder = os.path.expanduser('~')+'/Pictures/DustRadiativeForcing/lblrtm/'
#%% experimental setup for Solar Village
lats = np.array([24.907864502,])  # location & date to process
lons = np.array([46.397677359,])
date = dt.datetime(2007, 4, 9)  # the date of the flight close to Riyadh
date = dt.datetime(1991, 1, 2)  # TODO: use it temporary while ECMWF data is downloading
#%% setup atmospheric meteorological profile
atm_stag_ds = get_atmospheric_profile(date)  # stag indicates staggered profile
atm_stag_ds = atm_stag_ds.sel(lat=lats, lon=lons, time=date, method='nearest')  # TODO: date selection should be controlled by tollerance
atm_stag_ds['p'] = (('level', 'lat', 'lon'), atm_stag_ds['p'].data[:, np.newaxis, np.newaxis])  # keep in 1D although general approach is N-D
atm_stag_ds = atm_stag_ds.isel(level=range(len(atm_stag_ds.level) - 2))
#%% setup atmospheric chemical composition (gases). Setup H2O from atm grid (relative humidity)
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
#%% estimate LNFL & LBLRTM spectral requirements
'''
LNLFL: The TAPE3 file should include lines from at least 25 cm-1 on either end of the calculation region.
'''
print('LNFL TAPE5 range for SW should be: {} {}'.format(RRTM_SW_LW_WN_RANGE[0] - 25, RRTM_SW_LW_WN_RANGE[1] + 25))
#%% LBL: derive gases absorption OD & optical properties
dvout = 10
lblrtm_setup_vo = LblrtmSetup()
lblrtm_setup_vo.DVOUT = dvout  # cm^-1
lblrtm_setup_vo.zenithAngle = 180  # 0

# cross_sections = gases_ds.sel(species=[Gas.NO2.value, Gas.SO2.value])  # indicate the gases, for which the xsections should be accounted for
cross_sections = gases_ds.sel(species=[Gas.NO2.value, ])  # indicate the gases, for which the xsections should be accounted for
lblrtm_scratch_fp = '/Users/osipovs/Temp/'  # local
lblrtm_scratch_fp = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/run_tonga/'  # remote

fp = intermediate_data_storage_path + 'lblrtm_op_ds_w_rayleigh.nc'
if os.path.exists(fp):
    print('Reusing local copy of the previous LBLRTM w Rayleigh calculations')
    lblrtm_op_ds_w_rayleigh = xarray.open_dataset(fp)
else:
    print('\nPreparing LBLRTM run with Rayleigh scattering\n')
    lblrtm_op_ds_w_rayleigh = run_lblrtm_over_spectral_range(RRTM_SW_LW_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, lblrtm_setup_vo, True)
    lblrtm_op_ds_w_rayleigh.to_netcdf(fp)

fp = intermediate_data_storage_path + 'lblrtm_op_ds_wo_rayleigh.nc'
if os.path.exists(fp):
    print('Reusing local copy of the previous LBLRTM wo Rayleigh calculations')
    lblrtm_op_ds_wo_rayleigh = xarray.open_dataset(fp)
else:
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    lblrtm_op_ds_wo_rayleigh = run_lblrtm_over_spectral_range(RRTM_SW_LW_WN_RANGE, lblrtm_scratch_fp, atm_stag_ds, gases_ds_ref, cross_sections, lblrtm_setup_vo, False)
    lblrtm_op_ds_wo_rayleigh.to_netcdf(fp)
#%% derive the mixture optical properties + do the physical validation and fixes
print('Checking in: lblrtm_op_ds_wo_rayleigh')
checkin_and_fix(lblrtm_op_ds_wo_rayleigh)
print('Checking in: lblrtm_op_ds_w_rayleigh')
checkin_and_fix(lblrtm_op_ds_w_rayleigh)

rayleigh_op_ds = derive_rayleigh_optical_properties(lblrtm_op_ds_w_rayleigh, lblrtm_op_ds_wo_rayleigh)
print('Checking in: rayleigh_op_ds')
checkin_and_fix(rayleigh_op_ds)

mixed_op_ds_ref = mix_optical_properties([lblrtm_op_ds_wo_rayleigh, rayleigh_op_ds], externally=True)
print('Checking in: mixed_op_ds_ref')
checkin_and_fix(mixed_op_ds_ref)
#%% OPs: Mie & dust


def prepare_dust_sd_profile():
    sd_profile_df, column_sd_df, sd_columns, sd_diam_stag, sd_diam_rho = get_probed_size_distribution_in_riyadh(dt.datetime(2007, 4, 9))  # dNdlogD
    r_orig_rho = column_sd_df.index / 2
    r_mie_rho = np.logspace(-1, 2, 100)
    r_data = np.unique(np.concatenate((r_mie_rho, r_orig_rho)))
    ind = np.logical_and(r_data>=r_orig_rho.min(), r_data<=r_orig_rho.max())
    r_data = r_data[ind]

    # upsample
    f = sp.interpolate.interp1d(r_orig_rho, column_sd_df)
    dNdlogD = f(r_data)

    z_scale = 5*10**3  # m
    dNdlogD_profile = dNdlogD / z_scale  # number / cm^2 / m
    dNdlogD_profile *= 10**-8  # number / um^2 / m
    dNdlogD_profile = np.repeat(dNdlogD_profile[:, np.newaxis], 15, 1)
    sd_profile_ds = xr.Dataset(
        data_vars=dict(
            dNdlogD=(['radius', 'level'], dNdlogD_profile),
            z=(['level'], atm_stag_ds.z[0:15].data.squeeze()),
        ),
        coords=dict(
            radius=(['radius', ], r_data),
            level=(['level', ], atm_stag_ds.level[0:15].data.squeeze()),
        ),
        attrs=dict(description="Dust size distribution"),
    )

    return sd_profile_ds


sd_profile_ds = prepare_dust_sd_profile()  # uniform size distribution over 5 km
# to speed up calculations, use single layer for mie calculations


def prepare_dust_sd_profile_simplified():
    sd_profile_df, column_sd_df, sd_columns, sd_diam_stag, sd_diam_rho = get_probed_size_distribution_in_riyadh(dt.datetime(2007, 4, 9))  # dNdlogD
    r_orig_rho = column_sd_df.index / 2
    r_mie_rho = np.logspace(-1, 2, 100)
    r_data = np.unique(np.concatenate((r_mie_rho, r_orig_rho)))
    ind = np.logical_and(r_data>=r_orig_rho.min(), r_data<=r_orig_rho.max())
    r_data = r_data[ind]

    # upsample
    f = sp.interpolate.interp1d(r_orig_rho, column_sd_df)
    dNdlogD = f(r_data)

    z_scale = sd_profile_df.GALT.max()-sd_profile_df.GALT.min()  # 5*10**3  # m
    dNdlogD_profile = dNdlogD / z_scale  # number / cm^2 / m
    dNdlogD_profile *= 10**-8  # number / um^2 / m
    sd_profile_ds = xr.Dataset(
        data_vars=dict(
            dNdlogD=(['radius', ], dNdlogD_profile),
        ),
        coords=dict(
            radius=(['radius', ], r_data),
        ),
        attrs=dict(description="Dust size distribution"),
    )

    return sd_profile_ds, z_scale


sd_profile_ds, z_scale = prepare_dust_sd_profile_simplified()  # uniform size distribution over 5 km
ri_vo = get_dust_ri(wavelengths=rayleigh_op_ds.wavelength.data)

mie_file_path = intermediate_data_storage_path + 'mie_ds.nc'
dust_op_file_path = intermediate_data_storage_path + 'dust_op_ds.nc'
if os.path.exists(mie_file_path):
    print('Reusing local copy of the Mie calculations')
    mie_ds = xarray.open_dataset(mie_file_path)
    dust_op_ds = xarray.open_dataset(dust_op_file_path)
else:
    mie_ds = mie.get_mie_efficiencies(ri_vo['ri'], sd_profile_ds.radius.data, ri_vo['wl'])  # mie.
    mie_ds.to_netcdf(mie_file_path)

    dust_op_ds = mie.integrate_mie_over_aerosol_size_distribution(mie_ds, sd_profile_ds)
    dust_op_ds.to_netcdf(dust_op_file_path)
#%% scale SD to produce 0.5 OD at visible (By default it is around 1.5)
column_od = dust_op_ds.ext.sel(wavenumber=10**4/0.5).data * z_scale
dNdlogD_scale = 0.5 / column_od
dust_op_ds['ext'] *= dNdlogD_scale
sd_profile_ds['dNdlogD'] *= dNdlogD_scale
print('Scaling Dust SD to produce 0.5 column OD at 0.5 um ')
#%% CDFs: Compute the CDFs of the optical properties
dust_op_cdf_file_path = intermediate_data_storage_path + 'dust_op_cdf_ds.nc'
if os.path.exists(dust_op_cdf_file_path):
    print('Reusing local copy of the Dust OP CDF calculations')
    dust_op_cdf_ds = xarray.open_dataset(dust_op_cdf_file_path)
else:
    sd_cdf = []
    op_cdf = []
    for r_index, r in enumerate(sd_profile_ds.radius):
        print('Computing OP CDF for r {} / {}'.format(r_index, sd_profile_ds.radius.size))
        sd_subset_ds = sd_profile_ds.where(sd_profile_ds.radius <= r, 0)
        sd_cdf.append(sd_subset_ds)

        op_subset_ds = mie.integrate_mie_over_aerosol_size_distribution(mie_ds, sd_subset_ds)
        op_cdf.append(op_subset_ds)

    dust_op_cdf_ds = xr.concat(op_cdf, dim='radius')
    dust_op_cdf_ds['radius'] = sd_profile_ds.radius
    dust_op_cdf_ds.to_netcdf(dust_op_cdf_file_path)
#%% plot OD CDFs & print CDFs
print('Contribution of the dust particles with various radius to the Optical Properties.')

for wn in [10**4/0.5, 10**4/10]:
    fig, axes = plt.subplots(nrows=3, constrained_layout=True, figsize=(5,10))
    op_slice_ds = dust_op_cdf_ds.sel(wavenumber=wn)
    ax = axes[0]
    (op_slice_ds.ext/op_slice_ds.ext[-1]).plot(ax=ax)
    ax.set_xscale('log')
    ax = axes[1]
    (op_slice_ds.ssa/op_slice_ds.ssa[-1]).plot(ax=ax)
    ax.set_xscale('log')
    ax = axes[2]
    (op_slice_ds.g/op_slice_ds.g[-1]).plot(ax=ax)
    ax.set_xscale('log')
    plt.suptitle('Dust Optical Properties CDFs')
    save_figure(pics_folder, 'dust op cdfs {}'.format(wn))

    rs = [6, 10]
    for r_threshold in rs:
        op_r_slice_ds = op_slice_ds.sel(radius=r_threshold, method='nearest')
        print('\tWN {} cm^-1. Radius up to {:.2f} um to the:'.format(wn, op_r_slice_ds.radius))
        diag = op_r_slice_ds.ext / op_slice_ds.ext[-1]
        # print out the contribution of particles r>=10
        print('\t\tExtinction {:.2f}'.format(diag.data))
        diag = op_r_slice_ds.ssa / op_slice_ds.ssa[-1]
        print('\t\tSSA {:.2f}'.format(diag.data))
        diag = op_r_slice_ds.g / op_slice_ds.g[-1]
        print('\t\tg {:.2f}'.format(diag.data))


#%% Spectral plot to make sure everything is OK with the dust OP
fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
ax = axes[0,0]
dust_op_ds.ext.plot(ax=ax)
ax.set_xscale('log')
ax = axes[0,1]
dust_op_ds.ssa.plot(ax=ax)
ax.set_xscale('log')
ax = axes[1,0]
dust_op_ds.g.plot(ax=ax)
ax.set_xscale('log')
ax = axes[1,1]
dust_op_ds.sel(wavenumber=10 ** 4 / 0.5).phase_function.plot(ax=ax)
ax.set_yscale('log')
# polar plot is not very good because dust is strongly forward scattering
# ax=plt.subplot(224, projection='polar')
# ax.plot(op_ds_slice.angle, np.log(op_ds_slice.isel(wavenumber=0).phase_function))

plt.suptitle('Dust spectral OP\nColumn OD at 0.5 um is {}'.format(dust_op_ds.sel(wavenumber=10**4/0.5).ext.data*z_scale))
save_figure(pics_folder, 'dust spectral op')
#%% DISORT: run
disort_setup_vo = DisortSetup()
setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)

print('HARDCODING 0 zenith and azimuth angles')
disort_setup_vo.zenith_angle_degree = 0
disort_setup_vo.azimuth_angle_degree = 0

setup_surface_albedo(disort_setup_vo)
disort_setup_vo.albedo = 0.3  # desert albedo. 0.06 ocean albedo

wn_grid_step = rayleigh_op_ds.wavenumber[1] - rayleigh_op_ds.wavenumber[0]
disort_setup_vo.wn_grid_step = wn_grid_step.item()  # width of wn range in cm-1. From LBLRTM TODO: think how to immplement this fail safe

disort_output_file_path = intermediate_data_storage_path + 'disort_output_ds_ref.nc'
if os.path.exists(disort_output_file_path):
    print('Reusing local disort_output_ds_ref.nc')
    disort_output_ds_ref = xr.open_dataset(disort_output_file_path)
else:  # mixture of gas absorption & Rayleigh scattering
    disort_output_ds_ref = run_disort_spectral(mixed_op_ds_ref, atm_stag_ds, disort_setup_vo)
    disort_output_ds_ref.to_netcdf(disort_output_file_path)

# run the dust cases across the radii to generate the cdf of the forcing
disort_output_cdf_ds_file_path = intermediate_data_storage_path + 'disort_output_cdf_ds_dust.nc'
if os.path.exists(disort_output_cdf_ds_file_path):
    print('Reusing local disort_output_cdf_ds_dust.nc')
    disort_output_cdf_ds_dust = xr.open_dataset(disort_output_cdf_ds_file_path)
else:
    disort_cdfs = []
    radius_subgrid = dust_op_cdf_ds.radius[::10]  # coarsen the resolution to facilitate the calculatiobs
    radius_subgrid = xr.concat([radius_subgrid, dust_op_cdf_ds.sel(radius=[6,10], method='nearest').radius], dim='radius').sortby('radius')  # force to include 10 um
    for r_index, r in enumerate(radius_subgrid):
        print('Running DISORT r {} / {}'.format(r_index, radius_subgrid.size))
        # Expand these dust OPs in a 5 km layer and put zeroes above
        # each item here is the op derived from subset of dust dNdlr (cut up to certain size)
        aer_op_ds = dust_op_cdf_ds.sel(radius=r).expand_dims(dim={'level': rayleigh_op_ds.level.size})
        aer_op_ds['level'] = rayleigh_op_ds.level
        aer_op_ds['ext'] = aer_op_ds.ext.where(aer_op_ds.level > 550, 0)  # confine extinction to below 5km (1000-550 hPa)
        # derive OD
        dz = atm_stag_ds.z.diff(dim='level').squeeze()
        dz['level'] = aer_op_ds.ext.level
        aer_op_ds['od'] = (aer_op_ds.ext * dz)  # do not integrate.sum(dim='level')
        aer_op_ds = aer_op_ds.drop_vars(['ext', 'lat', 'lon', 'time', 'radius'])

        # get the mixtures of gas absorption & Rayleigh scattering + aerosol
        mixed_op_ds_dust = mix_optical_properties([lblrtm_op_ds_wo_rayleigh, rayleigh_op_ds, aer_op_ds], externally=True)
        mixed_op_ds_dust['wavelength'] = lblrtm_op_ds_wo_rayleigh.wavelength  # mixing looses wavelength for some reason. Restore it
        print('Checking in: mixed_op_ds_dust')
        checkin_and_fix(mixed_op_ds_dust)

        disort_output_ds_dust = run_disort_spectral(mixed_op_ds_dust, atm_stag_ds, disort_setup_vo)
        disort_cdfs.append(disort_output_ds_dust)

    disort_output_cdf_ds_dust = xr.concat(disort_cdfs, dim='radius')
    disort_output_cdf_ds_dust['radius'] = radius_subgrid
    disort_output_cdf_ds_dust.to_netcdf(disort_output_cdf_ds_file_path)

#%% PREP for Plotting: drop radiances to speed up things, declare lists
if 'radinaces' in disort_output_cdf_ds_dust.keys():
    disort_output_cdf_ds_dust = disort_output_cdf_ds_dust.drop_vars('radiances')
    disort_output_ds_ref = disort_output_ds_ref.drop_vars('radiances')

ds = disort_output_ds_ref
disort_keys = []
for key in ds.variables:
    if ds[key].ndim == 2:
        disort_keys += [key, ]

wn_range_labels = ['SW', 'LW', 'NET (SW+LW)']
wn_ranges = [slice(RRTM_LW_WN_RANGE[1], None), slice(0, RRTM_LW_WN_RANGE[1]), slice(None, None), ]
#%% PLOT: Spectral disort output
axes = plot_spectral_profiles(disort_output_ds_ref, disort_keys)
plt.suptitle('DISORT: Ref')
save_figure(pics_folder, 'profiles_disort_ref_spectral')

disort_output_ds_dust = disort_output_cdf_ds_dust.isel(radius=-1)
plot_spectral_profiles(disort_output_ds_dust, disort_keys)
plt.suptitle('DISORT: Dust')
save_figure(pics_folder, 'profiles_disort_dust_spectral')

plot_spectral_profiles(disort_output_ds_dust-disort_output_ds_ref, disort_keys)
plt.suptitle('DISORT: Dust-Ref')
save_figure(pics_folder, 'profiles_disort_pmc_spectral')
#%% PLOT: Broadband disort output
model_label = 'DISORT'

for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    plot_ref_perturbed_pmc(wn_range, wn_range_label, 'Dust', disort_output_ds_ref, disort_output_ds_dust,
                           disort_keys, model_label, pics_folder)

#%% plot forcing CDF
model_label = 'DISORT. Dust CDF RF(r)'

for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    # derive RF
    rf_ds = disort_output_cdf_ds_dust - disort_output_ds_ref
    rf_ds = rf_ds.sel(wavenumber=wn_range).integrate('wavenumber')  # integrate now to avoid issues with division by 0
    ds = rf_ds / rf_ds.isel(radius=-1)  # this will produce the cdf

    if (rf_ds.isel(radius=-1).direct_flux_down==0).any():  # then I will get NaN due to division by 0
        print('DFD has zeroes, replacing CDF with 0')
        ds['direct_flux_down'] = ds.direct_flux_down.fillna(0)

    # TOA
    axes = plot_spectral_profiles(ds.isel(level=-1), disort_keys, yincrease=True, yscale='linear', grid=True)
    plt.suptitle('{}. {}\nTOA'.format(model_label, wn_range_label))
    save_figure(pics_folder, 'rf_toa_{}'.format(wn_range_label))

    # BOA
    axes = plot_spectral_profiles(ds.isel(level=0), disort_keys, yincrease=True, yscale='linear', grid=True)
    plt.suptitle('{}. {}\nBOA'.format(model_label, wn_range_label))
    save_figure(pics_folder, 'rf_boa_{}'.format(wn_range_label))

    # dA
    dA_ds = rf_ds.isel(level=-1)-rf_ds.isel(level=0)  # Compute dA first, then compute CDF
    ds = dA_ds / dA_ds.isel(radius=-1)  # this will produce the cdf

    axes = plot_spectral_profiles(ds, disort_keys, yincrease=True, yscale='linear', grid=True)  # TOA-BOA = dA
    plt.suptitle('{}. {}\ndA'.format(model_label, wn_range_label))
    save_figure(pics_folder, 'rf_dA_{}'.format(wn_range_label))


#%% print the diags
print('Contribution of dust particles to the RF')
for wn_range, wn_range_label in zip(wn_ranges, wn_range_labels):
    print(wn_range_label)
    rs = [6, 10]
    for r_threshold in rs:
        print('radius up to {} um'.format(r_threshold))
        rf_ds = disort_output_cdf_ds_dust - disort_output_ds_ref
        rf_ds = rf_ds.sel(wavenumber=wn_range).integrate('wavenumber')  # integrate now to avoid issues with division by 0
        ds = rf_ds / rf_ds.isel(radius=-1)  # this will produce the cdf

        print('\tat TOA: {:.2f}'.format(ds.sel(radius=r_threshold, method='nearest').isel(level=-1).down_minus_up_flux.data))
        print('\tat BOA: {:.2f}'.format(ds.sel(radius=r_threshold, method='nearest').isel(level=0).down_minus_up_flux.data))

        dA_ds = rf_ds.isel(level=-1) - rf_ds.isel(level=0)  # Compute dA first, then compute CDF
        ds = dA_ds / dA_ds.isel(radius=-1)  # this will produce the cdf

        print('\tdA: {:.2f}'.format(ds.sel(radius=r_threshold, method='nearest').down_minus_up_flux.data))
