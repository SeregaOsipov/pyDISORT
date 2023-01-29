import numpy as np
import disort
from scipy import special
import xarray as xr


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
        self.PHI = 0


def run_disort_spectral(op_ds, atm_ds, disort_setup_vo):
    for wl_index in range(op_ds.wavelength.size):
        disort_output_item = run_disort(op_ds.isel(wavelength=wl_index), atm_ds, disort_setup_vo)

    disort_output_item

def run_disort(op_ds, atm_ds, disort_setup_vo):
    '''
    Monochromatic version (single wavelength)
    '''
    NLYR = op_ds.dims['level']
    DTAUC = op_ds.od
    SSALB = op_ds.ssa

    if np.any(np.isnan(DTAUC*SSALB)):
        raise Exception('DisortController:runDisortAtGivenWavelength, optical depth or ssa has NaN values')

    NSTR = disort_setup_vo.NSTR
    NMOM = disort_setup_vo.NMOM

    if NMOM < NSTR:
        raise Exception('DisortController:runDisortAtGivenWavelength, NMOM<NSTR, increase NMOM')
        NMOM = NSTR

    PMOM = np.zeros([NMOM+1, NLYR])  # TODO: check dimensions
    PMOM[:] = np.NaN
    pmoms = ()
    for momentIndex in range(NMOM+1):
        pmom_item = compute_phase_function_moments(op_ds.phase_function, momentIndex)
        pmoms += (pmom_item, )

    pmom = xr.concat(pmoms, dim='phase_function_moment')

    TEMPER = atm_ds.t
    USRTAU = False
    NTAU = 2
    UTAU = np.zeros([NTAU,])  # TODO: check dimensions
    UTAU[1-0] = 0.0
    UTAU[2-1] = np.max(DTAUC)

    USRANG = disort_setup_vo.USRANG
    NUMU = disort_setup_vo.NUMU
    UMU = disort_setup_vo.UMU
    if USRANG:
        if (UMU==0).any():  # UMU must NOT have any zero values
            raise('pyDISORT:run_disort. UMU must not have any zero values')

    NPHI = disort_setup_vo.NPHI
    PHI = disort_setup_vo.PHI

    IBCND = 0

    # I need proper beam intensity calculations here
    # [ zenithAngle, julianDay, totalSolarFlux, azimuthAngle, directIrradiance, diffuseIrradiance ] = computeSolarZenithAngle( currentDate, swInputSettingsVO.geoPointVO);
    # interpolate solar function to the requested wavelength or scale afterwards
    # solarIrradianceValue = interp1(this.solarIrrandianceWaveLengthData, this.solarIrradianceData, opticalPropertiesVO.waveLengthData)  # TODO: implement
    solarIrradianceValue = 1
    print('TODO: Implement TOA spectral Solar Constant')
    # display(['wl: ' num2str(opticalPropertiesVO.waveLengthData) ', irrad: ' num2str(solarIrradianceValue)])
    FBEAM = solarIrradianceValue
    UMU0 = np.cos(np.rad2deg(disort_setup_vo.zenith_angle_degree))
    PHI0 = disort_setup_vo.azimuth_angle_degree

    FISOT = 0

    LAMBER = True  # simple Lambertian reflectivity
    ALBEDO = disort_setp_vo.albedo

    BTEMP = atm_ds.skt
    TTEMP = atm_ds.t[-1]  # TODO: check index: 0 or end
    raise('TODO: check index: 0 or end')
    TEMIS = 0

    PLANK = False
    WVNMLO = 0
    WVNMHI = 50000

    ACCUR = 0.0  # should be between 0 and 0.1. I used to use single(0.005)
    NTEST = 0
    NPASS = 0

    ONLYFL = False
    PRNT = np.array([False, False, False, False, False])
    HEADER = 'pyDISORT header'  # use '' if crashes

    # DISORT LAYERING CONVENTION:  Layers are numbered from the top boundary down.
    raise('reverse the order of the profiles')
    # TODO: check if reversing is necessary
    DTAUC = flip(DTAUC)  # reverse the order of the profiles
    SSALB = flip(SSALB)
    PMOM = flip(PMOM,2)
    TEMPER = flip(TEMPER)

    DO_PSEUDO_SPHERE = disort_setup_vo.DO_PSEUDO_SPHERE  # setting to true fixed negative radiances
    DELTAMPLUS = disort_setup_vo.DELTAMPLUS  # requires NMOM = NSTR + 1;

    #%%
    H_LYR = np.zeros([NLYR + 1, ])
    RHOQ = np.zeros((int(NSTR / 2), int(NSTR / 2) + 1, NSTR))
    RHOU = np.zeros((NUMU, int(NSTR / 2) + 1, NSTR))
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
    # TODO:  restore the proper layers ordering, bottom to top
    # RFLDIR = flip(RFLDIR);  # restore the proper layers ordering, bottom to top
    # RFLDN = flip(RFLDN);
    # FLUP = flip(FLUP);
    # UAVG = flip(UAVG);
    # UU = flipdim(UU,2);

    disort_output_ds = None
    # disortOutputVO = DisortOutputVO();
    # disortOutputVO.directFluxDownData = RFLDIR;
    # disortOutputVO.diffuseFluxDownData = RFLDN;
    # disortOutputVO.fluxUpData = FLUP;
    # disortOutputVO.actinicFluxData = 4*pi*UAVG;  % UAVG is mean intensity, thus normalized over the entire sphere, multiple by 4*pi to get actinic flux
    # disortOutputVO.radiancesData = UU;

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