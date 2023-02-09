import os
import os.path

import numpy as np
import xarray
import xarray as xr
import datetime as dt

from lblrtm_utils import write_settings_to_tape, Gas, run_lblrtm, read_od_output, LblrtmSetup, run_lblrtm_over_spectral_range
from disort_utils import run_disort, DisortSetup, run_disort_spectral, setup_viewing_geomtry, \
    prep_sun_spectral_irradiance, setup_surface_albedo

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

# LBLRTM has a profile zmax at 42.906, which causes the profile adjustments. Stay below it for now
#sublevels = atm_ds.level[atm_ds.level>3]  # TODO: add ZMAX check
# sublevels = atm_ds.level[atm_ds.level>8]  # TODO: add ZMAX check
# atm_ds = atm_ds.sel(level=sublevels)
atm_stag_ds.z[0, 0, 0] = 0  # TODO: temp test is z has to start from zero
atm_stag_ds = atm_stag_ds.isel(level=range(len(atm_stag_ds.level) - 2))
#%% setup atmospheric chemical composition (gases)


def get_atmospheric_gases_composition():
    fp = '/work/mm0062/b302074/Data/NASA/GMI/gmiClimatology.nc'  # GMI climatology
    fp = os.path.expanduser('~') + '/Temp/gmiClimatology.nc'  # GMI climatology
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


gases_ds = get_atmospheric_gases_composition()
gases_ds = gases_ds.sel(lat=lats, lon=lons, time=date.month-1, method='nearest')  # do the selection
# interpolate gases on the LBLRTM vertical grid
gases_ds = gases_ds.interp(level=atm_stag_ds.level.data, kwargs={"fill_value": "extrapolate"})  # TODO: extrapolating blindly is bad
# Add H2O, already on the MERRA2 grid
h2o_ds = gases_ds.sel(species=Gas.O2.value)  # Use O2 to make a single gas copy as a template for SO2
h2o_ds.const[:] = atm_stag_ds.r[:].data  # units H

h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
h2o_ds['units'] = (('species',), ['H',])  # relative humidity
# TODO: MERRA2 has negative relative humidity values, fix them
gases_ds = xr.concat([gases_ds, h2o_ds], 'species')

#%% derive optical properties

if os.path.exists('ds_with_rayleigh.nc'):
    print('Reusing local copy of the previos LBLRTM calculations')
    ds = xarray.open_dataset('ds_with_rayleigh.nc')
    ds_without_Rayleigh = xarray.open_dataset('ds_without_rayleigh.nc')
else:
    print('Runing LBLRTM clean ')
    # cross_sections = gases_ds.sel(species=[Gas.NO2.value, Gas.SO2.value])  # indicate the gases, for which the xsections should be accounted for
    cross_sections = gases_ds.sel(species=[Gas.NO2.value, ])  # indicate the gases, for which the xsections should be accounted for

    lblrtm_scratch_fp = '/Users/osipovs/Temp/'  # local
    lblrtm_scratch_fp = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/run_tonga/'  # remote

    min_wl = 0.5  # um
    max_wl = 0.55  # um

    # calculate optical properties. Remember that LBLRTM represent rho layer and optical properties of these layers
    print('\nPreparing LBLRTM run with Rayleigh scattering\n')
    ds = run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, True)
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    ds_without_Rayleigh = run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, False)
    ds.to_netcdf('ds_with_rayleigh.nc')
    ds_without_Rayleigh.to_netcdf('ds_without_rayleigh.nc')

#%% run disort
disort_setup_vo = DisortSetup()
setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)
setup_surface_albedo(disort_setup_vo)
# disort_setup_vo.NPHI
# PHI = disort_setup_vo.PHI

# UMU0 = np.cos(np.rad2deg(disort_setup_vo.zenith_angle_degree))
# PHI0 = disort_setup_vo.azimuth_angle_degree

op_ds = ds
disort_output_ds = run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo)
#%% sun irradiance

#%%
# derive Rayleigh OD
rayleigh_od_da = ds.od-ds_without_Rayleigh.od
rayleigh_od_da.plot()
import matplotlib.pyplot as plt
plt.show()



#%% run disort


#%%
import matplotlib.pyplot as plt
plt.ion()

plt.contourf(ds.od.data)
plt.show()

rayleigh_od_da.plot()
ds.od.plot()

#%%

rayleigh_od_da[:,0].plot()
plt.show()

#%% prescribe lower boundary conditions (surface albedo)







#%% test timezones
#%% time zone

from timezonefinder import TimezoneFinder
tf = TimezoneFinder()  # reuse
tz = tf.timezone_at(lng=13.358, lat=52.5061)  # 'Europe/Berlin'
date.tzinfo
import pytz
timezone = pytz.timezone(tz)
date_w_tz = timezone.localize(date)