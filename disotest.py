import numpy as np
print(disort.__doc__)  # print list of functions


'''
This is a ported disotest.f90 (only a single 1a case).
'''

ACCUR = 0.0
NTEST = 0
NPASS = 0
#%% IF( DOPROB(1) )  THEN
USRTAU=True
USRANG=True
LAMBER =True
PLANK =False
ONLYFL =False
DO_PSEUDO_SPHERE =False
DELTAMPLUS =False

NSTR = 16
NLYR = 1
NMOM = NSTR
NTAU = 2
NUMU = 6
NPHI = 1
IBCND = 0

#%% allocate arrays
DTAUC=np.zeros(NLYR)
SSALB=np.zeros(NLYR)
PMOM = np.ones((NMOM+1, NLYR))
TEMPER=np.zeros(NLYR+1)
UTAU=np.zeros(NTAU)
UMU=np.zeros(NUMU)
PHI = np.zeros(NPHI)
H_LYR = np.zeros(NLYR+1)

RHOQ = np.zeros((int(NSTR/2), int(NSTR/2)+1, NSTR))
RHOU = np.zeros((NUMU, int(NSTR/2)+1, NSTR))
EMUST = np.zeros(NUMU)
BEMST = np.zeros(int(NSTR/2))
RHO_ACCURATE=np.zeros((NUMU, NPHI))

RFLDIR=np.zeros(NTAU)
RFLDN=np.zeros(NTAU)
FLUP=np.zeros(NTAU)
DFDT=np.zeros(NTAU)
UAVG=np.zeros(NTAU)
ALBMED = np.zeros(NUMU)
TRNMED = np.zeros(NUMU)
UU = np.zeros((NUMU, NTAU, NPHI))

PRNT = np.array([True, False, False, False, True])

# DO 10  ICAS = 1,6
ICAS=1
NPROB = 1
# IF ( ICAS.EQ.1 ) THEN

UMU0 = 0.1
PHI0 = 0.0

UMU[1-1] = -1.0
UMU[2-1] = -0.5
UMU[3-1] = -0.1
UMU[4-1] = 0.1
UMU[5-1] = 0.5
UMU[6-1] = 1.0

PHI[1-1] = 0.0
ALBEDO = 0.0

UTAU[1-1] = 0.0
UTAU[2-1] = 0.03125

#%%
PMOM=np.ones((NMOM+1, NLYR))  # NOTE that dim indexing starts from 0, i.e. PMOM( 0:NMOM, NLYR )
PMOM[:,0] = disort.getmom(1, 0, NMOM)  # getmom is only implemented for a single layer. Don't confuse local variables
#%%
DTAUC[1-1] = UTAU[2-1]
SSALB[1-1] = 0.2
FBEAM = np.pi / UMU0
FISOT = 0.0
#%% junk
ACCUR = np.NaN
BTEMP = np.NaN
TEMIS = np.NaN
TTEMP = np.NaN
WVNMLO = np.NaN
WVNMHI = np.NaN
EARTH_RADIUS = 6371.0
HEADER=''

#%%
RFLDIR, RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED = disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER, DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC, SSALB, PMOM, TEMPER, WVNMLO, WVNMHI, UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT, ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU, RHO_ACCURATE, BEMST, EMUST, ACCUR,  HEADER)

# prtfin is not implemented yet
#disort.prtfin(UTAU, NTAU, UMU, NUMU, PHI, NPHI, ONLYFL, RFLDIR, RFLDN, FLUP, DFDT, UU, disort.dochek.tstfir[1-1,ICAS-1,NPROB-1], disort.dochek.tstfdn[1-1,ICAS-1,NPROB-1], disort.dochek.tstfup[1-1,ICAS-1,NPROB-1], disort.dochek.tstdfd[1-1,ICAS-1,NPROB-1], disort.dochek.tstuu[1-1,1-1,1-1,ICAS-1,NPROB-1], NTEST, NPASS )