from climpy.utils.disort_utils import prep_chanceetal_sun_spectral_irradiance, get_aer_solar_constant
from scipy import integrate

'''
The script compares two sources of solar irradiance: AER and Chance et al.
'''

#%% prep datasets
chance_df = prep_chanceetal_sun_spectral_irradiance()  # up to 1 micron
aer_df = get_aer_solar_constant(in_wavelength_grid=True)  # entire SW and up to 12 um
# get the subset at chance wls
aer_subset_df = aer_df[(aer_df.index >= chance_df.index.min()) & (aer_df.index <= chance_df.index.max())]

#%%
integral_aer = aer_subset_df.apply(lambda g: integrate.trapz(g, x=g.index))
integral_chance = chance_df.apply(lambda g: integrate.trapz(g, x=g.index))
ratio = integral_aer/integral_chance
print('AER/Chance et al. ratio of solar constant is {}'.format(ratio.item()))
#%% Let's see how much solar energy sits outside 3-4 um
aer_solar_lw_subset_df = aer_df[(aer_df.index >= 4)]
integral_lw = aer_solar_lw_subset_df.apply(lambda g: integrate.trapz(g, x=g.index))
