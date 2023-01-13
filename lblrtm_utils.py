import subprocess
import os
from enum import Enum
import struct
import numpy as np
import xarray as xr


class Gas(Enum):
    H2O = 'H2O'
    CO2 = 'CO2'
    O3 = 'O3'
    N2O = 'N2O'
    CO = 'CO'
    CH4 = 'CH4'
    O2 = 'O2'
    NO = 'NO'
    SO2 = 'SO2'
    NO2 = 'NO2'


# this gases comes in order from Table 2 of the rrtm_sw_instructions file, it is NOT a full list, only first part of it
AER_SUPPORTED_GASES = [Gas.H2O, Gas.CO2, Gas.O3, Gas.N2O, Gas.CO, Gas.CH4, Gas.O2, Gas.NO, Gas.SO2, Gas.NO2]


class LblrtmSetup(object):  # TODO: Temp dummy settings
    pass


def write_settings_to_tape(tape_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, include_Rayleigh_extinction):
    '''
    Convention is to treat reanalysis (MERRA2) output as a staggered grid.

    :param tape_fp: output file path
    :param atm_ds:
    :param gases_ds: GMI output climatology  as DataArray
    :param include_Rayleigh_extinction:
    :return:
    '''
    IHIRAC = 1  #1 - Voigt profile
    ILBLF4 = 0  # 1

    ICNTNM = 5  # all continua calculated, except Rayleigh extinction
    if include_Rayleigh_extinction:
        ICNTNM = 1

    IAERSL = 0
    IEMIT = 0  # optical     depth    only
    ISCAN = 0
    IFILTR = 0
    IPLOT = 0
    ITEST = 0
    IATM = 1
    IMRG = 1
    ILAS = 0
    IOD = 1  # 0
    IXSECT = 0
    if len(cross_sections.species) > 0:
        IXSECT = 1
    MPTS = 0
    NPTS = 0

    # IHIRAC, ILBLF4, ICNTNM, IAERSL,  IEMIT,  ISCAN, IFILTR, IPLOT, ITEST,  IATM,  IMRG,  ILAS,   IOD, IXSECT,  MPTS,  NPTS
    # 5,     10,     15,     20,     25,     30,     35,    40,    45,    50, 54-55,    60,    65,     70, 72-75, 77-80
    # 4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1,  4X,I1, 4X,I1, 4X,I1, 4X,I1, 3X,A2, 4X,I1, 4X,I1,  4X,I1, 1X,I4, 1X,I4

    #%% output section
    tape = open(tape_fp, 'w+')
    tape.write('%c%s\n' % ('$', 'SW Era Interim, GMI, MODIS'))
    tape.write('%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n' % (IHIRAC, ILBLF4, ICNTNM, IAERSL, IEMIT, ISCAN,
            IFILTR, IPLOT, ITEST, IATM, IMRG, ILAS, IOD, IXSECT, MPTS, NPTS, MPTS, NPTS))

    if IHIRAC + IAERSL + IEMIT + IATM + ILAS > 0:
        V1 = lblrtm_setup_vo.V1  # remember V2 has to be bigger than V1
        V2 = lblrtm_setup_vo.V2
        if V2 - V1 > 2020:
            raise Exception('LBLRTM output: spectral range is wider than 2020 cm^-1')
        if V2 < V1:
            raise Exception('LBLRTM output: V2<V1, you probably dont want it')

        SAMPLE = 4
        DVSET = 0  # 0.000007
        ALFAL0 = 0.04
        AVMASS = 36
        DPTMIN = 0.0002  # 0
        DPTFAC = 0.001  # 0
        ILNFLG = 0
        DVOUT = lblrtm_setup_vo.DVOUT
        NMOL_SCAL = 0

        # V1,     V2,   SAMPLE,   DVSET,  ALFAL0,   AVMASS,   DPTMIN,   DPTFAC,   ILNFLG,     DVOUT,   NMOL_SCAL
        # 1-10,  11-20,    21-30,   31-40,   41-50,    51-60,    61-70,    71-80,     85,      90-100,         105
        # E10.3,  E10.3,    E10.3,   E10.3,   E10.3,    E10.3,    E10.3,    E10.3,    4X,I1,  5X,E10.3,       3x,I2
        tape.write('%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%5d%5s%10.3E%5d\n' % (V1, V2, SAMPLE, DVSET, ALFAL0, AVMASS, DPTMIN, DPTFAC, ILNFLG, ' ', DVOUT, NMOL_SCAL))

    if IEMIT == 1:
        raise Exception('LBLRTM output: IEMIT=1 is not implemented and should not be')

        TBOUND = 300
        SREMIS = []
        SREMIS[1-1] = 0
        SREMIS[2-1] = 0
        SREMIS[3-1] = 0

        SRREFL = []
        SRREFL[1-1] = 0
        SRREFL[2-1] = 0
        SRREFL[3-1] = 0

        surf_refl = 'l'

        # TBOUND, SREMIS(1), SREMIS(2), SREMIS(3), SRREFL(1), SRREFL(2), SRREFL(3), surf_refl
        # 1-10,     11-20,     21-30,     31-40,     41-50,     51-60,     61-70,    75
        # E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3,     E10.3    4X,1A
        # looks like I have to replace it with separate file with spectral properties
        tape.write('%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%10.3E%5s\n', TBOUND, SREMIS(1), SREMIS(2), SREMIS(3), SRREFL(1), SRREFL(2), SRREFL(3), surf_refl);

    MODEL = 0
    ITYPE = 2

    IBMAX = -1 * len(atm_ds.p)  # stag grid. TOTAL NUMBER OF LAYERS
    print('{LBLRTMParser:outputSettings} Running in the z profile grid instead of p')
    IBMAX = len(atm_ds.p)

    ZERO = 2
    NOPRNT = 0
    NMOL = len(gases_ds.species)  # NUMBER OF    MOLECULAR    SPECIES
    IPUNCH = 1  # TAPE7 FILE OF MOLECULAR COLUMN AMOUNTS FROM LBLATM ONLY FOR IATM=1; IPUNCH=1 (CARD 3.1)
    IFXTYP = 0
    MUNITS = 0
    RE = 6371.23  # km
    HSPACE = 0  # atm_ds.z[-1] / 10 ** 3  # altitude definition for space
    VBAR = 0
    REF_LAT = atm_ds.lat  # TODO: add check, has to have a single coordinate

    # MODEL, ITYPE, IBMAX, ZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS, RE, HSPACE, VBAR, REF_LAT
    # 5, 10, 15, 20, 25, 30, 35, 36 - 37, 39 - 40, 41 - 50, 51 - 60, 61 - 70, 81 - 90
    # I5,     I5,    I5,      I5,      I5,    I5,     I5,     I2,   1X, I2, F10.3,  F10.3, F10.3,   10x, F10.3
    tape.write('%5d%5d%5d%5d%5d%5d%5d%2d%3d%10.3f%10.3f%10.3f%10s%10.3f\n' % (MODEL, ITYPE, IBMAX, ZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS, RE, HSPACE, VBAR, ' ', REF_LAT))

    # for ITYPE = 2, only 3 of the first 5 parameters are required to specify the path, e.g., H1, H2, ANGLE or H1, H2 and RANGE
    # for ITYPE = 3, H1 = observer altitude must be specified.Either H2 = tangent height or ANGLE must be specified.Other parameters are ignored.
    # just test to see which one is higher, H1 or H2
    H1 = atm_ds.z[-1] / 10 ** 3  # observer altitude
    H2 = atm_ds.z[0] / 10 ** 3  # for ITYPE = 2, H2 is the end point altitude
    #H2 = atm_ds.z_sfc  # try surface height instead, since z_sfc is not included in the profile
    if IBMAX < 0:
        H1 = atm_ds.p[-1]
        H2 = atm_ds.p[0]
        #H2 = atm_ds.sp  # again, use surface instead of profile

    ANGLE = lblrtm_setup_vo.zenithAngle  # zenith angle at H1 (degrees)
    # ANGLE = swInputVO.zenithAngle + 90; % zenith
    RANGE = 0  # H2-H1; %length of a straight path from H1 to H2 (km)
    BETA = 0  # earth centered angle from H1 to H2 (degrees)
    LEN = 0
    HOBS = 0  # Height of observer, used only for informational purposes in satellite-type simulations when computing output geometry above 120 km.

    # H1, H2, ANGLE, RANGE, BETA, LEN, HOBS
    # 1 - 10, 11 - 20, 21 - 30, 31 - 40, 41 - 50, 51 - 55, 61 - 70
    # F10.3, F10.3,   F10.3,   F10.3,  F10.3,    I5, 5X,F10.3
    tape.write('%10.3f%10.3f%10.3f%10.3f%10.3f%5d%5s%10.3f\n' % (H1, H2, ANGLE, RANGE, BETA, LEN, ' ', HOBS))

    for j in range(abs(IBMAX)):
        format = '%10.3f'
        if j % 8 == 7:
            format = '%10.3f\n'
        if IBMAX > 0:
            tape.write(format % (atm_ds.z[j] / 10 ** 3))
        else:
            tape.write(format % atm_ds.p[j])

    tape.write('\n')

    IMMAX = len(atm_ds.p)  # number of atmospheric profile boundaries
    IMMAX = IBMAX
    HMOD = 'profile description'
    # IMMAX, HMOD
    # 5, 6 - 29
    # I5, 3A8
    tape.write('%5d%s\n' % (IMMAX, ' '))

    write_gases_to_tape(atm_ds, gases_ds, tape, IBMAX)  # TODO: previous implementation is missing JLONG variable

    if IXSECT > 0:  # write cross-sections
        IXMOLS = cross_sections.species.size
        IPRFL = 0
        IXSBIN = 0
        tape.write('%5d%5d%5d\n' % (IXMOLS, IPRFL, IXSBIN))
        for specie in cross_sections.species:
            gas_ds = cross_sections.sel(species=specie)
            format = '%10s'
            if j % 8 == 7:
                format = '%10s\n'
            tape.write(format % specie.item())
        tape.write('\n')

        LAYX = gas_ds.level.size
        IZORP = 0  # 0 - km, 1 - hPa
        # Sync cross-sections output with the user atmosphjere
        if IBMAX < 0:  # then atmospheric profile is in p.
            IZORP = 1
        XTITLE = 'major UV absorpers'
        tape.write('%5d%5d%50s\n' % (LAYX, IZORP, XTITLE))

        # assume that all the species has the same vertical grid
        for layer_index in range(LAYX):
            if IZORP==0:  # z, km
                tape.write('%10.3f%5s' % (atm_ds.z[layer_index], ''))
            else: # pressure, hPa
                tape.write('%10.3f%5s' % (gas_ds.level[layer_index], ''))
            # output units first
            for specie in cross_sections.species:
                gas_ds = cross_sections.sel(species=specie)
                tape.write('%1s' % 'A')
                if gas_ds.units.item() != 'A':
                    raise Exception('LBLRTM crosssections: LBL only accepts 1 or A units here')
            tape.write('\n')
            # now output densities
            for specie in cross_sections.species:
                gas_ds = cross_sections.sel(species=specie)
                format = '%10.3E'
                if j % 8 == 7:
                    format = '%10.3E\n'
                tape.write(format % gas_ds.const[layer_index])
            tape.write('\n')

    tape.write('%%')
    tape.close()


def write_gases_to_tape(atm_ds, gases_ds, tape, ibmax):
    '''

    :param atm_ds:
    :param gases_ds:
    :param tape: is a file handle
    :param ibmax:
    :return:
    '''

    # preprocessing (possible could be done using xarray's sortby
    ordered_gases = []  # sort gases in order required by model
    for specie_enum in AER_SUPPORTED_GASES:
        gas_ds = gases_ds.sel(species=specie_enum.value)  # missing gas will raise an exception
        # raise Exception('RRTMParser outputGasProperties: reguired gas {} is missing'.format(RRTMAbstractParser.rrtmSupportedGasesEnums[i].name))
        ordered_gases += [gas_ds, ]

    # TODO: check that gases have been interpolated on the LBLRTM vertical grid

    # ZM,    PM,    TM,    JCHARP, JCHART,   JLONG,   (JCHAR(M),M =1,39)
    # 1-10, 11-20, 21-30,        36,     37,      39,     41  through  80
    # E10.3, E10.3, E10.3,   5x,  A1,     A1,  1x, A1,     1x,    39A1

    for k in range(abs(ibmax)):
        tape.write('%10.3E' % (atm_ds.z[k] / 10**3,))
        tape.write('%10.3E' % atm_ds.p[k])
        tape.write('%10.3E' % atm_ds.t[k])

        tape.write('%6c%c' % ('A', 'A'))
        tape.write('%2c' % 'L')
        tape.write('%c' % ' ')

        for gas_ds in ordered_gases:
            tape.write('%c' % gas_ds.units.item())

        # tape.write('\n');
        for i, gas_ds in enumerate(ordered_gases):
            if i % 8 == 0:  # add \n every 8 items
                tape.write('\n')
            tape.write('%15.8E' % gas_ds.const[k])
        tape.write('\n')


def run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, include_Rayleigh_extinction):
    '''
    Desc
    :param lblrtm_scratch_fp:
    :param lblrtm_setup_vo:
    :param atm_ds:
    :param gases_ds:
    :param cross_sections:
    :param include_Rayleigh_extinction:
    :return:

    Run example:

    lblrtm_setup_vo = LblrtmSetup()
    lblrtm_setup_vo.V1 = 2000
    lblrtm_setup_vo.V2 = 4000
    lblrtm_setup_vo.DVOUT = 10
    lblrtm_setup_vo.zenithAngle = 0

    tape5_fp = '{}{}'.format(lblrtm_scratch_fp, 'TAPE5')
    # write_settings_to_tape(tape5_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, True)
    run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, True)

    spectral_od_profile, wavelengts, wavenumbers = read_od_output(lblrtm_scratch_fp, atm_ds.level.size)
    '''

    lblExecutablePath = '/work/mm0062/b302074/workspace/fortran/AER-RC/LBLRTM/lblrtm_v12.11_linux_intel_dbl'
    lblExecutablePath = '/home/osipovs/Temp/lblrtm_v12.11_linux_intel_dbl'  # TODO: this is just a copy from levante. Recompile locally

    # TODO: always remove prevous files before reruning
    print('TODO: always remove prevous files before reruning !!!')
    #'\rm ODdef* TAPE3? TAPE6? TAPE?? TAPE7 TAPE9 TAPE5'
    #'\rm fort.601 fort.602 fort.603'

    # output TAPE5
    tape5_fp = '{}{}'.format(lblrtm_scratch_fp, 'TAPE5')
    write_settings_to_tape(tape5_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, include_Rayleigh_extinction)

    postfixString = ''
    subprocess.run([lblExecutablePath, postfixString], cwd=lblrtm_scratch_fp)


def run_lblrtm_over_spectral_range(min_wl, max_wl, lblrtm_scratch_fp, atm_ds, gases_ds, cross_sections, include_Rayleigh_extinction=False):
    '''
    Wavelengths in um.

    :param min_wl:
    :param max_wl:
    :param lblrtm_scratch_fp:
    :param lblrtm_setup_vo:
    :param atm_ds:
    :param gases_ds:
    :param cross_sections:
    :param include_Rayleigh_extinction:
    :return:
    '''

    max_wn = 10**4 / min_wl
    min_wn = 10**4 / max_wl
    wn_step = 2000  # 1010  # max wn width is 2000 in LBLRTM

    wn_grid = np.arange(min_wn, max_wn, wn_step)
    if wn_grid[-1] != max_wn:  # last element is included in numpy arange
        wn_grid = np.concatenate([wn_grid, [max_wn]])

    lblrtm_setup_vo = LblrtmSetup()
    lblrtm_setup_vo.DVOUT = 10  # cm^-1
    lblrtm_setup_vo.zenithAngle = 180  # 0

    print('Running LBLRTM over spectral range')
    ods = []
    wls = []
    for index in range(len(wn_grid)-1):
        lblrtm_setup_vo.V1 = wn_grid[index]
        lblrtm_setup_vo.V2 = wn_grid[index+1]

        print('Spectral interval {}/{}: {} to {} cm^-1'.format(index+1, len(wn_grid)-1, lblrtm_setup_vo.V1, lblrtm_setup_vo.V2))

        run_lblrtm(lblrtm_scratch_fp, lblrtm_setup_vo, atm_ds, gases_ds, cross_sections, include_Rayleigh_extinction)
        spectral_od_profile, wavelengts, wavenumbers = read_od_output(lblrtm_scratch_fp, atm_ds.level.size)

        ods += [spectral_od_profile, ]
        wls += [wavelengts, ]

    spectral_od_profile = np.concatenate(ods, axis=1)
    wavelengts = np.concatenate(wls, axis=0)

    # convert into full set of optical properties of gas mixture
    od = spectral_od_profile
    ssa = np.zeros(od.shape)
    g = np.zeros(od.shape)
    phase_function = np.zeros(od.shape + (100,))
    phase_function_angles = np.linspace(0, np.pi, phase_function.shape[-1])

    ds = xr.Dataset(
        data_vars=dict(
            od=(["level", "wavelength"], od),
            ssa=(["level", "wavelength"], ssa),
            g=(["level", "wavelength"], g),
            phase_function=(["level", "wavelength", "angle"], phase_function),
        ),
        coords=dict(
            level=(['level', ], atm_ds.level.data),
            wavelength=(['wavelength', ], wavelengts),
            angle=(['angle', ], phase_function_angles),
        ),
        attrs=dict(description="Optical properties according to LBLRTM"),
    )

    return ds

# def run_spectral_lblrtm(inputSettingsVO, minWavelength, maxWavelength, include_Rayleigh_extinction=False):
#     '''
#
#     :param inputSettingsVO:
#     :param minWavelength: in microns
#     :param maxWavelength: in microns
#     :param include_Rayleigh_extinction:
#     :return:
#     '''
#
#     maxWavenumber = 10 ^ 4 / minWavelength;
#     minWavenumber = 10 ^ 4 / maxWavelength;
#
#     # lblrtm has limit on max wavenumber width (2000), so I have to split it into several intervals
#     # for a different spectral ranges either LBLRTM or internal reader breaks down
#     waveNumberStep = 2000
#     waveNumberStep = 1010
#
#     waveNumberGrid = minWavenumber:waveNumberStep: maxWavenumber
#     if (maxWavenumber > max(waveNumberGrid)):
#         waveNumberGrid = cat(2, waveNumberGrid, maxWavenumber)
#
#     spectralOpticalDepthProfileData = [];
#     waveLengthData = [];
#
#     for i=1:numel(waveNumberGrid) - 1
#         inputSettingsVO.V1 = waveNumberGrid(i)
#         inputSettingsVO.V2 = waveNumberGrid(i + 1)
#         inputSettingsVO.DVOUT = 100; % cm ^ -1
#         inputSettingsVO.DVOUT = 10; % cm ^ -1
#         % inputSettingsVO.DVOUT = 1; % cm ^ -1
#
#         inputSettingsVO.zenithAngle = 0  # it is important to specify 0 zenith angle to get normal optical depths, otherwise LBLRTM will compute slant path
#         display(['covering spectral range from ' num2str(10 ^ 4 / inputSettingsVO.V2) ' to ' num2str(10 ^ 4 / inputSettingsVO.V1)])
#
#         current_work_dir = cd(this.scratchFilePath)
#         run_lblrtm(inputSettingsVO, include_Rayleigh_extinction)
#
#         [spectralOpticalDepthProfileItemData, waveLengthItemData] = this.lblOutputParser.parseProfileOutput(numel(inputSettingsVO.layersPressures))
#         spectralOpticalDepthProfileData = cat(1, spectralOpticalDepthProfileData, spectralOpticalDepthProfileItemData)
#         waveLengthData = cat(2, waveLengthData, waveLengthItemData)
#         cd(current_work_dir)
#
#     gasOpticalPropertiesVO = OpticalPropertiesVO();
#     gasOpticalPropertiesVO.odData = spectralOpticalDepthProfileData;
#     gasOpticalPropertiesVO.ssaData = zeros(size(gasOpticalPropertiesVO.odData));
#     gasOpticalPropertiesVO.gData = zeros(size(gasOpticalPropertiesVO.odData));
#     numberOfAngles = 100;
#     pfSize = cat(2, size(gasOpticalPropertiesVO.odData), numberOfAngles);
#     gasOpticalPropertiesVO.phaseFunctionData = zeros(pfSize);
#     gasOpticalPropertiesVO.phaseFunctionAnglesData = linspace(0, pi, numberOfAngles);
#     gasOpticalPropertiesVO.waveLengthData = waveLengthData;
#     gasOpticalPropertiesVO.layersPressureData = inputSettingsVO.layersPressures;
#     gasOpticalPropertiesVO.boundariesPressureData = inputSettingsVO.boundariesPressures;


def red_tape11_output(tape_fp, opt):
    '''
    % File format illustration
    % for single precision
    % shift 266*4 bytes
    % LOOP
    % 1 int        , 24 (block of v1, v2, dv, npts)
    % 2 double vars, for v1, and v2
    % 1 float      , for dv
    % 1 int        , for npts
    % 1 int        , 24
    % 1 int        , 9600 or npts*4 (beg of block output)
    % NPTs float   , rad
    % 1 int        , 9600 or npts*4 (end of block of output)
    % LOOP ENDS

    % for double precision
    % shift 356*4 bytes
    % LOOP
    % 1 int        , 32 (v1, v2, dv and npts, extra 0)
    % 3 double vars, for v1, v2, and dv
    % 1 long int   , for npts
    % 1 int        , 32
    % 1 int        , 19200 or npts*8 (beg of block of output)
    % NPTS double  , rad
    % 1 int        , 19200 or npts*8 (end of block of output)
    % LOOP ENDS

    % Author: Xianglei Huang
    % Tested on Redhat Linux with pgi-compiler version of LBLRTM
    % ported by Sergey Osipov from Matlab to Python
    '''

    v = []
    rad = []

    fid = open(tape_fp, mode='rb')

    if opt[0].lower() == 'f' or opt[0].lower() == 's':
        shift = 266
        itype = 1
    else:
        shift = 356
        itype = 2

    fid.seek(shift*4)
    test = struct.unpack('i', fid.read(4))

    little_endian = False  # in most cases it is a little endian.
    if (itype == 1 and test == 24) or (itype == 2 and test == 32):
        little_endian = True
    # else big endian

    endflg = 0
    panel = 0

    if itype == 1:
        while endflg == 0:
            raise('Not ported completely')
            panel += 1
            # disp(['read panel ', int2str(panel)])
            v1, = struct.unpack('d', fid.read(8))  # 1, 'double');
            if isnan(v1):
                break
            v2, = struct.unpack('d', fid.read(8))  # 1, 'double');
            dv, = struct.unpack('f', fid.read(4))  # 1, 'float');
            npts, = struct.unpack('i', fid.read(4))  # 1, 'int');
            fid.read(4)

            LEN, = struct.unpack('i', fid.read(4))
            if LEN != 4 * npts:
                raise('internal file inconsistency')
                endflg = 1
            tmp, = struct.unpack('f'*npts, fid.read(4*npts))  # , npts, 'float');
            LEN2, = struct.unpack('i', fid.read(4))
            if LEN != LEN2:
                raise('internal file inconsistency')
                endflg = 1
            v += [v1, v2, dv]  # this concatenation is probably in wrong dimensions, check
            rad += tmp
    else:
        while endflg == 0:
            panel += 1
            # disp(['read panel ', int2str(panel)]);
            v1, v2, dv = struct.unpack('ddd', fid.read(8*3))  # 3, 'double');
            if np.isnan(v1):
                break
            npts, = struct.unpack('q', fid.read(8))  # 1, 'int64');  # q or Q

            if npts != 2400:
                endflg = 1

            struct.unpack('i', fid.read(4))  # 1, 'int')
            LEN, = struct.unpack('i', fid.read(4))  # 1, 'int')
            if LEN != 8 * npts:
                raise('internal file inconsistency')  # or print
                endflg = 1

            tmp = struct.unpack('d'*npts, fid.read(8*npts))  # npts, 'double')
            LEN2, = struct.unpack('i', fid.read(4))
            if LEN != LEN2:
                raise('internal file inconsistency')
                endflg = 1

            v += [v1, v2, dv]  # this concatenation is probably in wrong dimensions, check
            rad += tmp
    fid.close()
    return np.array(v), np.array(rad)


def read_od_output(lblrtm_scratch_fp, n_layers_in_profile):

    spectral_items = []
    for layer_index in range(n_layers_in_profile):
        # TODO: it is possible that layer does not exist because it was zeroed out by LBLRTM
        v, spectral_item = red_tape11_output('{}/ODint_{:03d}'.format(lblrtm_scratch_fp, layer_index+1), 'double')
        v1 = v[0]
        v2 = v[1]
        dv = v[2]
        spectral_items += [spectral_item,]

    spectral_od_profile = np.array(spectral_items)
    wavenumbers = np.linspace(v1, v2, spectral_item.shape[0])
    wavelengts = 10** 4. / wavenumbers

    return spectral_od_profile, wavelengts, wavenumbers

