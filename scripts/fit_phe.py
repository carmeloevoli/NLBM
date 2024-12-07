import numpy as np
from iminuit import Minuit
import math

from fit_BC import experiment_chi2

import constants

def model_primary(energy, params):
    xi, alpha, lnEb, dalpha = params 
    s = 0.025
    tauG = 1. * constants.MYR
    value = constants.SN_RATE * tauG / constants.GALACTIC_VOLUME # cm^-3
    value *= (xi / 100.) * constants.E_CR / np.power(constants.E_0, 2.) * (alpha - 2.) # GeV^-1 cm^-3
    value *= constants.C_LIGHT / 4. / math.pi # GeV^-1 cm^-2 s^-1 sr^-1
    Eb = np.exp(lnEb)
    y = value * np.power(energy / constants.E_0, -alpha)
    y *= np.power(1. + np.power(energy / Eb, dalpha / s), s)
    return y * 1e4 # GeV^-1 m^-2 s^-1 sr^-1

def fit_protons(initial_params):
    def chi2_function(xi_H, alpha_H, xi_He, alpha_He, lnEb, dalpha):
        chi2 = 0.
        H_files = ['../data/AMS-02_H_Ek.txt', '../data/CALET_H_Ek.txt', '../data/DAMPE_H_Ek.txt']
        for filename in H_files:
            chi2 += experiment_chi2(model_primary, filename, [xi_H, alpha_H, lnEb, dalpha], min_energy=50.)
        He_files = ['../data/AMS-02_He_Ekn.txt', '../data/CALET_He_Ekn.txt', '../data/DAMPE_He_Ekn.txt']
        for filename in He_files:
            chi2 += experiment_chi2(model_primary, filename, [xi_He, alpha_He, lnEb + np.log(2), dalpha], min_energy=50.)
        return chi2

    """Perform the chi-squared fit for the B/C model."""
    # Perform minimization
    m = Minuit(chi2_function, xi_H=initial_params[0], alpha_H=initial_params[1], 
               xi_He=initial_params[2], alpha_He=initial_params[3], 
               lnEb=initial_params[4], dalpha=initial_params[5])
    m.errordef = Minuit.LEAST_SQUARES

    # Optimize using simplex and migrad algorithms
    m.simplex()
    m.migrad()
    m.minos()
    
    # Print fit results
    print(f'xi_H      : {m.values[0]:.1f}')
    print(f'alpha_H   : {m.values[1]:.2f}')
    print(f'xi_He     : {m.values[2]:.2f}')
    print(f'alpha_He  : {m.values[3]:.2f}')
    print(f'E_b    : {np.exp(m.values[4]):.0f}')
    print(f'dalpha : {m.values[5]:.2f}')
    print(f'[{m.values[0]:.3f}, {m.values[1]:.3f}, {m.values[2]:.3f}, {m.values[3]:.3f}, {m.values[4]:.3f}, {m.values[5]:.3f}]')
    dof = 1.  # degrees of freedom placeholder
    return m.values, m.errors, m.fval, dof

def fit_helium(initial_params):
    def chi2_function(xi, alpha, lnEb, dalpha):
        chi2 = 0.
        He_files = ['../data/AMS-02_He_Ekn.txt', '../data/CALET_He_Ekn.txt', '../data/DAMPE_He_Ekn.txt']
        for filename in He_files:
            chi2 += experiment_chi2(model_primary, filename, [xi, alpha, lnEb, dalpha], min_energy=50.)
        return chi2
    
    """Perform the chi-squared fit for the B/C model."""
    # Perform minimization
    m = Minuit(chi2_function, xi=initial_params[0], alpha=initial_params[1], lnEb=initial_params[2], dalpha=initial_params[3])
    m.errordef = Minuit.LEAST_SQUARES

    # Optimize using simplex and migrad algorithms
    m.simplex()
    m.migrad()
    m.minos()
    
    # Print fit results
    print(f'xi     : {m.values[0]:.1f}')
    print(f'alpha  : {m.values[1]:.2f}')
    print(f'E_b    : {np.exp(m.values[2]):.0f}')
    print(f'dalpha : {m.values[3]:.2f}')
    print(f'[{m.values[0]:.3f}, {m.values[1]:.3f}, {m.values[2]:.3f}, {m.values[3]:.3f}]')
    dof = 1.  # degrees of freedom placeholder
    return m.values, m.errors, m.fval, dof

if __name__ == "__main__":
    # Initial guess for the parameters
    initial_guess = [2., 2.8, 0.15, 2.68, np.log(400.), 0.25]
    fit_protons(initial_guess)
    #initial_guess = [0.15, 2.680, np.log(800.), 0.193]
    #fit_helium(initial_guess)