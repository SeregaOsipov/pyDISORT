import os
import os.path

import numpy as np
import xarray
import xarray as xr
import datetime as dt

from climpy.utils.optics_utils import derive_rayleigh_optical_properties
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
atm_stag_ds['p'] = (('level', 'lat', 'lon'), atm_stag_ds['p'].data[:, np.newaxis, np.newaxis])  # keep in 1D although general approach is N-D

# atm_stag_ds.z[0, 0, 0] = 0  # TODO: temp test is z has to start from zero
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


gases_ds = get_atmospheric_gases_composition()
gases_ds = gases_ds.sel(lat=lats, lon=lons, time=date.month-1, method='nearest')  # do the selection
# interpolate gases on the LBLRTM vertical grid
gases_ds = gases_ds.interp(level=atm_stag_ds.level.data, kwargs={"fill_value": "extrapolate"})  # TODO: extrapolating blindly is bad
# Add H2O, already on the MERRA2 grid
h2o_ds = gases_ds.sel(species=Gas.O2.value).copy(deep=True)  # Use O2 to make a single gas copy as a template for SO2

# h2o_ds.const[:] = atm_stag_ds.r[:].data  # units H # TODO: MERRA2 has negative relative humidity values, fix them
h2o_ds.const[:] = rel_humidity_to_mass_concentration(atm_stag_ds) * 10**3  # A-ppmv, D-mass density (gm m-3)

h2o_ds['species'] = (('species',), [Gas.H2O.value, ])
# h2o_ds['units'] = (('species',), ['H',])  # relative humidity
h2o_ds['units'] = (('species',), ['D',])  # mass density (gm m-3)

gases_ds = xr.concat([gases_ds, h2o_ds], 'species')

gases_ds_ref = gases_ds

#%% prep the perturbation experiment
gases_ds_tonga = gases_ds_ref.copy(deep=True)
gases_ds = gases_ds_tonga

h2o_ds = gases_ds.sel(species=Gas.H2O.value)
levels = h2o_ds['const'].level
h2o_ds['const'].loc[dict(level=levels[levels<100])] *= 1.1  # boost water vapor above 100 hpa by 10%

#%% derive optical properties

if os.path.exists('op_ds_w_rayleigh.nc'):
    print('Reusing local copy of the previos LBLRTM calculations')
    op_ds_w_rayleigh = xarray.open_dataset('op_ds_w_rayleigh.nc')
    op_ds_wo_rayleigh = xarray.open_dataset('op_ds_wo_rayleigh.nc')
    op_ds_tonga = xarray.open_dataset('op_ds_tonga.nc')
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
    op_ds_w_rayleigh = run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, True)
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    op_ds_wo_rayleigh = run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, False)
    print('\nPreparing LBLRTM run for Tonga perturbation\n')
    op_ds_tonga = run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_stag_ds, gases_ds_tonga, cross_sections, False)

    op_ds_w_rayleigh.to_netcdf('op_ds_w_rayleigh.nc')
    op_ds_wo_rayleigh.to_netcdf('op_ds_wo_rayleigh.nc')
    op_ds_tonga.to_netcdf('op_ds_tonga.nc')


#%% run disort
rayleigh_op_ds = derive_rayleigh_optical_properties(op_ds_w_rayleigh, op_ds_wo_rayleigh)

disort_setup_vo = DisortSetup()
setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)
setup_surface_albedo(disort_setup_vo)

op_ds = op_ds_w_rayleigh
disort_output_ds = run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo)



#%%
import matplotlib.pyplot as plt
plt.ion()
plt.cla()
h2o_ds.const.plot()
#disort_output_ds.direct_flux_down.plot()
#%% sun irradiance

#%%
# derive Rayleigh OD
rayleigh_od_da = op_ds_w_rayleigh.od - op_ds_wo_rayleigh.od
rayleigh_od_da.plot()
import matplotlib.pyplot as plt
plt.show()



#%% run disort


#%%
import matplotlib.pyplot as plt
plt.ion()

plt.contourf(op_ds_w_rayleigh.od.data)
plt.show()

rayleigh_od_da.plot()
op_ds_w_rayleigh.od.plot()

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