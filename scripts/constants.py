import math

# UNITS
YEAR = 3.154e7  # s
MYR = 1e6 * YEAR  # s
PC = 3.086e18  # cm
KPC = 1e3 * PC  # cm
GEV_TO_ERG = 0.00160218  # erg
MBARN = 1e-27  # cm^2
ERG_TO_GEV = 624.151

# PHYSICAL CONSTANTS
C_LIGHT = 3e10  # cm/s
PROTON_MASS = 1.67262192e-24  # g
MP_C2 = 0.938  # GeV
ME_C2 = 0.511e-3  # GeV
SIGMA_THOMSON = 6.6524e-25  # cm^2
U_CMB = 0.25e-9  # GeV / cm^3

# FIXED PARAMETERS
SN_RATE = 1. / 50 / YEAR  # s^-1
DISK_HEIGHT = 100. * PC  # cm
GALACTIC_RADIUS = 10. * KPC  # cm
GALACTIC_VOLUME = math.pi * GALACTIC_RADIUS**2 * 2. * DISK_HEIGHT  # cm^3
E_0 = 10.  # GeV
E_CR = 1e51 * ERG_TO_GEV  # GeV

# CROSS-SECTIONS
F_HE = 1.4
SIGMA_OC = 60. * MBARN  # cm^2
SIGMA_OB = 37. * MBARN  # cm^2
SIGMA_CB = 71. * MBARN  # cm^2
#N_H = 0.5  # cm^-3