import os
import os.path

import numpy as np
import xarray
import xarray as xr
import datetime as dt

from climpy.utils.atmos_utils import get_atmospheric_profile
from climpy.utils.lblrtm_utils import Gas, run_lblrtm_over_spectral_range, LblrtmSetup
from climpy.utils.disort_utils import DisortSetup, run_disort_spectral, setup_viewing_geomtry, \
    setup_surface_albedo

#%% experimental setup
lats = np.array([-20.55,])  # location & date to process
lons = np.array([175.385,])
date = dt.datetime(1991, 1, 2)
#%% setup atmospheric meteorological profile
atm_stag_ds = get_atmospheric_profile(date)  # stag indicates staggered profile
atm_stag_ds = atm_stag_ds.sel(lat=lats, lon=lons, time=date, method='nearest')  # TODO: date selection should be controlled by tollerance

# atm_stag_ds.z[0, 0, 0] = 0  # TODO: temp test is z has to start from zero
atm_stag_ds = atm_stag_ds.isel(level=range(len(atm_stag_ds.level) - 2))
#%% setup atmospheric chemical composition (gases)
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
dvout = 10
lblrtm_setup_vo = LblrtmSetup()
lblrtm_setup_vo.DVOUT = dvout  # cm^-1
lblrtm_setup_vo.zenithAngle = 180  # 0

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
    wn_range = [10**4/max_wl, 10**4/min_wl]

    # calculate optical properties. Remember that LBLRTM represent rho layer and optical properties of these layers
    print('\nPreparing LBLRTM run with Rayleigh scattering\n')
    ds = run_lblrtm_over_spectral_range(wn_range, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, lblrtm_setup_vo, True)
    print('\nPreparing LBLRTM run without Rayleigh scattering\n')
    ds_without_Rayleigh = run_lblrtm_over_spectral_range(wn_range, lblrtm_scratch_fp, atm_stag_ds, gases_ds, cross_sections, lblrtm_setup_vo, False)
    ds.to_netcdf('ds_with_rayleigh.nc')
    ds_without_Rayleigh.to_netcdf('ds_without_rayleigh.nc')

#%% run disort
disort_setup_vo = DisortSetup()
setup_viewing_geomtry(disort_setup_vo, lats[0], lons[0], date)
setup_surface_albedo(disort_setup_vo)

op_ds = ds
disort_output_ds = run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo)

#%% prep the perturbattion experiment


#%%
import matplotlib.pyplot as plt
plt.ion()
disort_output_ds.direct_flux_down.plot()
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