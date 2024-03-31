import numpy as np
from iminuit import Minuit
import math
from dataclasses import dataclass
from scipy.integrate import quad

# UNITS
year = 3.154e7 # s
Myr = 1e6 * year # s
pc = 3.086e18 # cm
kpc = 1e3 * pc # cm
GeV = 0.00160218 # erg
mbarn = 1e-27 # cm2
erg2GeV = 624.151

# PHYSICAL CONSTANTS
c_light = 3e10 # cm s-1
proton_mass = 1.67262192e-24 # gr
mp_c2 = 0.938 # GeV
me_c2 = 0.511e-3 # GeV
sigma_th = 6.6524e-25 # cm2
U_CMB = 0.25e-9 # GeV / cm3

# FIXED PARAMS
sn_rate = 1. / 50 / year # s-1
h_disk = 100. * pc # cm
R_G = 10. * kpc # cm
V_G = math.pi * np.power(R_G, 2.) * 2. * h_disk # cm3
E_0 = 10. # GeV
E_CR = 1e51 * erg2GeV # GeV
sigma_CB = 61. * mbarn # cm^2
sigma_OB = 35. * mbarn # cm^2
sigma_OC = 27. * mbarn # cm^2
sigma_C10 = 3.8 * mbarn # cm^2
sigma_C9 = 5.7 * mbarn # cm^2
sigma_O10 = 2.4 * mbarn # cm^2
sigma_O9 = 3.9 * mbarn # cm^2
n_H = 0.5 # cm^-3

@dataclass
class GrammageParams:
    XG: float
    Xc: float
    zeta: float
    fCO: float
    
@dataclass
class InjectionParams:
    xi: float
    lnEb: float
    alpha: float
    dalpha: float
    s: float
    tauG: float

def model_BC(E, par):
    X_cr = sigma_CB / proton_mass + par.fCO * sigma_OB / proton_mass
    value_G = X_cr * par.XG
    value_c = X_cr * par.Xc * np.power(E, -par.zeta * np.log(E))
    return value_G + value_c

def model_CO(E, par):
    X_cr = sigma_OC / proton_mass
    value_G = X_cr * par.XG
    value_c = X_cr * par.Xc * np.power(E, -par.zeta * np.log(E))
    return par.fCO + (value_G + value_c)
    
def model_Be_ratio(E, tau_G, f_OC):
    gamma_Be = (E + mp_c2) / mp_c2
    tau_d = 2. * Myr * gamma_Be
    sigma_10 = sigma_C10 + f_OC * sigma_O10
    sigma_9 = sigma_C9 + f_OC * sigma_O9
    value = sigma_10 / sigma_9 * tau_d / (tau_G + tau_d)
    return value

def model_primary(E, par, doBreak = True):
    value = sn_rate * par.tauG / V_G # cm^-3
    value *= par.xi * E_CR / np.power(E_0, 2.) * (par.alpha - 2.) # GeV^-1 cm^-3
    value *= c_light / 4. / math.pi # GeV^-1 cm^-2 s^-1 sr^-1
    Eb = np.exp(par.lnEb)
    y = value * (E / E_0)**(-par.alpha)
    if doBreak:
        y *= np.power(1. + np.power(E / Eb, par.dalpha / par.s), par.s)
    return y * 1e4 # GeV^-1 m^-2 s^-1 sr^-1

def model_antiprotons(E, grammage_params, H_params, He_params, doBreak = True):
    from get_xsecs import dsigmadE_ap

    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = dsigmadE_ap('cubic')

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        return Eprime * grammage_params.XG * (model_primary(Eprime, H_params, doBreak) * pp_s + model_primary(Eprime, He_params, doBreak) * hep_s)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        X_c = grammage_params.Xc * np.power(Eprime, -grammage_params.zeta * np.log(Eprime))
        return Eprime * X_c * (model_primary(Eprime, H_params, doBreak) * pp_s + model_primary(Eprime, He_params, doBreak) * hep_s)

    size = len(E)
    
    value_G = np.zeros(size)
    for i in range(size):
        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e3 * E[i]), args=(E[i]))[0]
    value_G /= proton_mass
    
    value_c = np.zeros(size)
    for i in range(size):
        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e3 * E[i]), args=(E[i]))[0]
    value_c /= proton_mass
    
    return value_G, value_c
    
def test_xsecs(E, slope):
    from get_xsecs import dsigmadE_ap

    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = dsigmadE_ap('cubic')

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        flux = np.power(Eprime, -slope)
        return Eprime * flux * pp_s

    size = len(E)
    
    y = np.zeros(size)
    for i in range(size):
        y[i] = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]

    return np.power(E, slope) * y

def model_positrons(E, grammage_params, H_params, He_params, doBreak = True):
    from get_xsecs import dsigmadE_pos

    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = dsigmadE_pos('cubic')

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        return Eprime * grammage_params.XG * (model_primary(Eprime, H_params, doBreak) * pp_s + model_primary(Eprime, He_params, doBreak) * hep_s)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        X_c = grammage_params.Xc * np.power(Eprime, -grammage_params.zeta * np.log(Eprime))
        return Eprime * X_c * (model_primary(Eprime, H_params, doBreak) * pp_s + model_primary(Eprime, He_params, doBreak) * hep_s)

    size = len(E)
    
    value_G = np.zeros(size)
    for i in range(size):
        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e3 * E[i]), args=(E[i]))[0]
    value_G /= proton_mass
    
    value_c = np.zeros(size)
    for i in range(size):
        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e3 * E[i]), args=(E[i]))[0]
    value_c /= proton_mass
    
    y = compute_tauLoss(E) / (compute_tauG(grammage_params) + compute_tauLoss(E))
    
    return y * value_G, y * value_c
    
def compute_tauLoss(E):
    gamma_e = E / me_c2
    U = U_CMB
    value = 3. / 4. * me_c2 / c_light / sigma_th / U / gamma_e
    return value
    
def compute_tauG(par):
    tau = par.XG / (proton_mass * c_light * n_H)
    return tau
    
def fit_BC(par):
    def experiment_chi2(filename, par, doBC):
        xd, yd, errd_lo, errd_up = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, errd_lo, errd_up):
            if doBC:
                Y = model_BC(x_i, par)
            else:
                Y = model_CO(x_i, par)
            if Y > y_i:
                chi2 += np.power((Y - y_i) / err_up_i, 2.)
            else:
                chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2
 
    def chi2_function(XG, Xc, zeta, fCO):
        chi2 = 0.
        chi2 += experiment_chi2('data/AMS-02_BC.txt', GrammageParams(XG, Xc, zeta, fCO), True)
        chi2 += experiment_chi2('data/CALET_BC.txt', GrammageParams(XG, Xc, zeta, fCO), True)
        chi2 += experiment_chi2('data/DAMPE_BC.txt', GrammageParams(XG, Xc, zeta, fCO), True)
        chi2 += experiment_chi2('data/AMS-02_CO.txt', GrammageParams(XG, Xc, zeta, fCO), False)
        chi2 += experiment_chi2('data/CALET_CO.txt', GrammageParams(XG, Xc, zeta, fCO), False)
        return chi2

    m = Minuit(chi2_function, XG=par.XG, Xc=par.Xc, zeta=par.zeta, fCO=par.fCO)
    m.errordef = Minuit.LEAST_SQUARES
    #m.fixed['fCO'] = True
 
    m.simplex()
    m.migrad()
    m.minos()

    #print(m.params)
    
    E = np.logspace(1, 5, 1000)
    values = GrammageParams(m.values[0], m.values[1], m.values[2], m.values[3])
    y = model_BC(E, values)
    
    X_cr = sigma_CB / proton_mass + values.fCO * sigma_OB / proton_mass
    y_G = X_cr * values.XG

    print(f'X_G  : {values.XG:5.2f} gr/cm2')
    print(f'X_0  : {values.Xc:5.2f} gr/cm2')
    print(f'zeta : {values.zeta:5.2f}')
    print(f'fCO  : {values.fCO:5.2f}')

    return E, y, y_G, values

def fit_H(par):
    def experiment_chi2(filename, par, range):
        x, y, err_tot_lo, err_tot_up = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(x, y, err_tot_lo, err_tot_up):
            if x_i > range[0] and x_i < range[1]:
                Y = model_primary(x_i, par)
                if Y > y_i:
                    chi2 += np.power((Y - y_i) / err_up_i, 2.)
                else:
                    chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2
 
    def chi2_function(xi, lnEb, alpha, dalpha, s, tauG):
        chi2 = 0.
        chi2 += experiment_chi2('data/AMS-02_H.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        chi2 += experiment_chi2('data/DAMPE_H.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        chi2 += experiment_chi2('data/CALET_H.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        return chi2

    m = Minuit(chi2_function, xi=par.xi, lnEb=par.lnEb, alpha=par.alpha, dalpha=par.dalpha, s=par.s, tauG=par.tauG)
    m.errordef = Minuit.LEAST_SQUARES
    m.fixed['tauG'] = True
    m.limits['lnEb'] = (4., 7.)
    #m.limits['s'] = (0.01, 1.)
    m.fixed['s'] = True

    m.simplex()
    m.migrad()
#    m.minos()

    #print(m.params)
    
    E = np.logspace(1, 9, 1000)
    values = InjectionParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5])
    y = model_primary(E, values)

    print(f'H_xi = {values.xi:5.3f}')
    print(f'H_Eb = {np.exp(values.lnEb):5.3f} GeV')
    print(f'H_alpha = {values.alpha:5.3f}')
    print(f'H_daplha = {values.dalpha:5.3f}')
    print(f'H_s = {values.s:5.3f}')
    
    return E, y, values

def fit_He(par):
    def experiment_chi2(filename, par, range):
        x, y, err_tot_lo, err_tot_up = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(x, y, err_tot_lo, err_tot_up):
            if x_i > range[0] and x_i < range[1]:
                Y = model_primary(x_i, par)
                if Y > y_i:
                    chi2 += np.power((Y - y_i) / err_up_i, 2.)
                else:
                    chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2

    def chi2_function(xi, lnEb, alpha, dalpha, s, tauG):
        chi2 = 0.
        chi2 += experiment_chi2('data/AMS-02_He.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        chi2 += experiment_chi2('data/DAMPE_He.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        chi2 += experiment_chi2('data/CALET_He.txt', InjectionParams(xi, lnEb, alpha, dalpha, s, tauG), [30., 1e4])
        return chi2

    m = Minuit(chi2_function, xi=par.xi, lnEb=par.lnEb, alpha=par.alpha, dalpha=par.dalpha, s=par.s, tauG=par.tauG)
    m.errordef = Minuit.LEAST_SQUARES
    m.fixed['tauG'] = True
    m.limits['lnEb'] = (4., 7.)
    m.fixed['s'] = True
    #m.limits['s'] = (0.01, 1.)

    m.simplex()
    m.migrad()
#    m.minos()

    #print(m.params)

    E = np.logspace(1, 9, 1000)
    values = InjectionParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5])
    y = model_primary(E, values)
 
    print(f'He_xi = {values.xi:5.3f}')
    print(f'He_Eb = {np.exp(values.lnEb):5.3f} GeV')
    print(f'He_alpha = {values.alpha:5.3f}')
    print(f'He_daplha = {values.dalpha:5.3f}')
    print(f'He_s = {values.s:5.3f}')

    return E, y, values
