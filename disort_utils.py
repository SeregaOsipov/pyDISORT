import numpy as np
import disort
import pandas as pd
from scipy import special
import xarray as xr
import pvlib
import scipy as sp

class DisortSetup(object):
    '''
    Resembles the DISORT settings preserving the input naming
    '''

    def __init__(self, NSTR=16):
        self.NSTR = NSTR
        self.NMOM = NSTR + 1
        self.DO_PSEUDO_SPHERE = True
        self.DELTAMPLUS = True  # requires NMOM = NSTR + 1

        self.USRANG = False
        self.NUMU = NSTR
        self.UMU = None

        self.NPHI = 1
        self.PHI = np.zeros((self.NPHI,))


def run_disort_spectral(op_ds, atm_ds, disort_setup_vo):

    def get_TOA_spectral_irradiance():
        sun_spectral_irradiance_df = prep_sun_spectral_irradiance()
        f = sp.interpolate.interp1d(sun_spectral_irradiance_df.index, sun_spectral_irradiance_df['irradiance'])
        irradiance_I = f(op_ds.wavelength)
        sun_spectral_irradiance_df = pd.Series(irradiance_I, index=op_ds.wavelength)
        return sun_spectral_irradiance_df

    sun_spectral_irradiance_df = get_TOA_spectral_irradiance()  # interpolate solar function to the requested wavelength or scale afterwards

    spectral_list = []
    for wl_index in range(op_ds.wavelength.size):
        disort_setup_vo.FBEAM = sun_spectral_irradiance_df.iloc[wl_index]
        disort_output_ds = run_disort(op_ds.isel(wavelength=wl_index), atm_ds, disort_setup_vo)
        spectral_list += [disort_output_ds, ]

    disort_spectral_output_ds = xr.concat(spectral_list, dim='wavelength')

def run_disort(op_rho_ds, atm_stag_ds, disort_setup_vo):
    '''
    Monochromatic version (single wavelength)
    op_rho_ds is the LBLRTM output on the RHO grid
    atm_stag_ds is (usually MERRA2) atmospheric profile on STAGGERED grid
    '''

    NLYR = op_rho_ds.dims['level']
    DTAUC = op_rho_ds.od
    SSALB = op_rho_ds.ssa

    if np.any(np.isnan(DTAUC*SSALB)):
        raise Exception('DisortController:runDisortAtGivenWavelength, optical depth or ssa has NaN values')

    NSTR = disort_setup_vo.NSTR
    NMOM = disort_setup_vo.NMOM

    if NMOM < NSTR:
        raise Exception('DisortController:runDisortAtGivenWavelength, NMOM<NSTR, increase NMOM')
        NMOM = NSTR

    pmoms = ()
    for momentIndex in range(NMOM+1):
        pmom_item = compute_phase_function_moments(op_rho_ds.phase_function, momentIndex)
        pmoms += (pmom_item, )
    pmom = xr.concat(pmoms, dim='phase_function_moment')

    TEMPER = atm_stag_ds.t  # at stag grid
    USRTAU = False
    NTAU = NLYR+1
    UTAU = np.zeros((NTAU,))  # Unsued (USRTAU is false), but have to initialize for F2PY dimensions logic

    USRANG = disort_setup_vo.USRANG
    NUMU = disort_setup_vo.NUMU
    UMU = disort_setup_vo.UMU
    if USRANG:  # UMU has to be initialized
        if (UMU==0).any():  # UMU must NOT have any zero values
            raise('pyDISORT:run_disort. UMU must not have any zero values')
    elif UMU is None:
        UMU = np.zeros((NUMU,))  # I have to have this line to initialize dimensions in DISORT right. Otherwise F2py struglles (which can be fixed probably)
    elif UMU.shape != (NUMU,):  # In this case UMU should not be specifed (since USRANG is false).
        raise Exception('disort_utils:run_disort. Incosistent UMU shape. UMU should not be specified at all')

    NPHI = disort_setup_vo.NPHI  # should be after the USRANG, but I need NPHI
    PHI = disort_setup_vo.PHI

    IBCND = 0
    FBEAM = disort_setup_vo.FBEAM
    UMU0 = np.cos(np.rad2deg(disort_setup_vo.zenith_angle_degree))  # Corresponding incident flux is UMU0 times FBEAM.
    PHI0 = disort_setup_vo.azimuth_angle_degree

    FISOT = 0

    LAMBER = True  # simple Lambertian reflectivity
    ALBEDO = disort_setup_vo.albedo

    BTEMP = atm_stag_ds.skt
    TTEMP = atm_stag_ds.t[atm_stag_ds.level == atm_stag_ds.level.min()]  # TOA or min p. Equivalent to atm_ds.t[-1]
    TEMIS = 0

    PLANK = False
    WVNMLO = 0  # only if PLANK
    WVNMHI = 50000

    ACCUR = 0.0  # should be between 0 and 0.1. I used to use single(0.005)
    NTEST = 0
    NPASS = 0

    ONLYFL = False
    # PRNT = np.array([True, False, False, False, True])  # more vervose
    PRNT = np.array([False, False, False, False, False])
    HEADER = 'pyDISORT header'  # use '' if crashes

    # DISORT LAYERING CONVENTION:  Layers are numbered from the top boundary down.
    DTAUC = DTAUC.sel(level=slice(None, None, -1))  # reverse the order of the profiles
    SSALB = SSALB.sel(level=slice(None, None, -1))
    PMOM = pmom.sel(level=slice(None, None, -1))
    TEMPER = TEMPER.sel(level=slice(None, None, -1))
    TEMPER = TEMPER.squeeze()

    DO_PSEUDO_SPHERE = disort_setup_vo.DO_PSEUDO_SPHERE  # setting to true fixed negative radiances
    DELTAMPLUS = disort_setup_vo.DELTAMPLUS  # requires NMOM = NSTR + 1;

    #%%
    H_LYR = np.zeros([NLYR + 1, ])
    RHOQ = np.zeros((int(NSTR/2), int(NSTR/2) + 1, NSTR))
    RHOU = np.zeros((NUMU, int(NSTR/2) + 1, NSTR))
    EMUST = np.zeros(NUMU)
    BEMST = np.zeros(int(NSTR / 2))
    RHO_ACCURATE = np.zeros((NUMU, NPHI))
    EARTH_RADIUS = 6371.0

    #%%
    RFLDIR, RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED = disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK,
                                                                        LAMBER, DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC,
                                                                        SSALB, PMOM, TEMPER, WVNMLO, WVNMHI, UTAU, UMU0,
                                                                        PHI0, UMU, PHI, FBEAM, FISOT, ALBEDO, BTEMP,
                                                                        TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                                                                        RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER)

    #%%
    RFLDIR = np.flipud(RFLDIR)  # restore the layers ordering, bottom to top
    RFLDN = np.flipud(RFLDN)
    FLUP = np.flipud(FLUP)
    UAVG = np.flipud(UAVG)
    UU = np.flip(UU,axis=1)

    # new dims order: level, polar agnels, azimuthal_ange
    UU = np.swapaxes(UU, 0, 1)

    # inject the wavelength dimension
    disort_output_ds = xr.Dataset(
        data_vars=dict(
            direct_flux_down=(["level", "wavelength",], RFLDIR[:,np.newaxis]),
            diffuse_flux_down=(["level", "wavelength",], RFLDN[:,np.newaxis]),
            diffuse_flux_up=(["level", "wavelength",], FLUP[:,np.newaxis]),  # up is only diffuse
            #  UAVG is mean intensity, thus normalized over the entire sphere, multiply by 4*pi to get actinic flux
            actinic_flux=(["level", "wavelength",], 4*np.pi*UAVG[:,np.newaxis]),
            # Do not confuse computational polar angles (cos of) with Phase Function angles in op_rho_ds
            # UMU should hold the computational angles, but they are not returned at the moment and require editing disort.pyf (f2py)
            radiances=(["level", 'radiance_cos_of_polar_angles', 'radiance_azimuthal_angle', 'wavelength'], UU[..., np.newaxis]),  # Azimuthal angles is in degree (PHI)
        ),
        coords=dict(
            level=(["level", ], atm_stag_ds.level.data),
            wavelength=(["wavelength", ], np.array((op_rho_ds.wavelength.item(0),))), #
            radiance_cos_of_polar_angles=(["radiance_cos_of_polar_angles", ], UMU),  # the cosines of the computational polar angles
            radiacens_azimuthal_angle=(["radiance_azimuthal_angle", ], PHI),  # Azimuthal output angles (in degrees) # PHI
        ),
        attrs=dict(description="pyDISORT output"),
    )

    return disort_output_ds


def compute_phase_function_moments(phase_function_df, n):
    '''
    expansion of phase function is given by:
    function P(u) for each band is defined as: P(u) = sum over streams l { (2l+1) (PHASE_l) (P_l(u)) }
    where
    u = cos(theta)
    PHASE_l = the lth moment of the phase function
    P_l(u) = lth Legendre polynomial,

      phaseFunction is a function of (angle, layer)
    '''

    # if x.size != size(phaseFunction,ndims(phaseFunction)) )
    #     ME = MException('OpticsController:computePhaseFunctionMoments', 'dimensions do not agree');
    #    throw(ME);

    # def legendre(n, x):
    #     res = []
    #     for m in range(n + 1):
    #         res.append(special.lpmv(m, n, x))
    #     return np.array(res)

    x = np.cos(phase_function_df.angle)

    # associatedLegPol = legendre(n, x)
    # pmom = np.trapz(phase_function_df * associatedLegPol, x)
    associatedLegPol = special.lpmv(0, n, x)
    phase_function_df['cos(angle)'] = np.cos(phase_function_df.angle)
    integrand = phase_function_df*associatedLegPol
    pmom = integrand.integrate('cos(angle)')

    # TODO: the rest needs to be tested, especially with aerosols
    print('TODO: compute_phase_function_moments needs to be tested, especially with aerosols')

    pmom *= -1  # invert sign because integration is wrong way
    # don't forget coefficient in front of integral
    # pmom = (2*n+1)/2 * pmom
    # add given the specific expansion, RRTM moved the (2n+1) from the coefficient to the expansion
    pmom = 1/2 * pmom

    if n==0:  # % first moment has to be 1. if the PH is entirely 0 ( I think it is unphysical), then I get zeros. Fix it
        pmom[:] = 1

    if (np.abs(pmom) > 1).any():
       raise('compute_phase_function_moments: pmom magnitude error')

    return pmom


def setup_viewing_geomtry(disort_setup_vo, lat, lon, date):
    '''
    Sun viewing geometry. Options are pvlib and dnppy

    :param disort_setup_vo:
    :param lat:
    :param lon:
    :param date: use UTC time
    :return:
    '''

    # TODO: supply altitude
    # TODO: have to specify time zone: , tz=site.tz)
    # TODO: make it work with the set of coordinates
    solpos = pvlib.solarposition.get_solarposition(date, lat, lon, 0)

    disort_setup_vo.zenith_angle_degree = solpos['zenith'].iloc[0]
    disort_setup_vo.azimuth_angle_degree = solpos['azimuth'].iloc[0]


def prep_sun_spectral_irradiance():
    # https://www.sciencedirect.com/science/article/pii/S0022407310000610
    file_path = '/work/mm0062/b302074/Data/Harvard/SAO2010_solar_spectrum/sao2010.solref.converted.txt'
    delimiter = ' ';

    df = pd.read_table(file_path, skiprows=range(4), header=None, delim_whitespace=True, usecols=[0,2], index_col=0, names=['wavelength', 'irradiance'])
    df.index *= 10**-3  # um  # 'wavelength'
    df['irradiance'] *= 10**3  # W/(m2 um)
    return df


def setup_surface_albedo(disort_setup_vo):
    disort_setup_vo.albedo = 0