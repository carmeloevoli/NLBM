import numpy as np
from iminuit import Minuit
import math

import constants

def grammage_HM(energy, A, params):
    XG0, delta, Eb, fCO = params
    ddelta, s = delta, 0.1 
    XG = XG0 * np.power(energy / 10., -delta) 
    XG *= np.power(1. + np.power(energy / Eb, ddelta / s), s)
    sigma_in_A = 45. * constants.MBARN * np.power(A, 2. / 3.) # To Be Checked
    Xcr = constants.PROTON_MASS / sigma_in_A
    factor_G = XG / (1. + XG / Xcr)
    return factor_G

def model_BC_HM(energy, params):
    """Calculate the model for B/C."""
    XG0, delta, Eb, fCO = params
    factor = constants.SIGMA_CB / constants.PROTON_MASS + constants.SIGMA_OB / constants.PROTON_MASS / fCO
    return factor * grammage_HM(energy, 10., params)

def model_CO_HM(energy, params):
    """Calculate the model for C/O."""
    XG0, delta, Eb, fCO = params
    factor = constants.SIGMA_OC / constants.PROTON_MASS
    return fCO + factor * grammage_HM(energy, 12., params)

def X2tau(X):
    # n_G = 0.5 # cm-3
    # factor = constants.PROTON_MASS * constants.C_LIGHT * n_G
    # return X / factor
    return 0.

def load_data(filename, min_energy, max_energy=1e20):
    """Load data from file, filtering by energy range."""
    data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5), unpack=True)
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = data
    mask = (E > min_energy) & (E < max_energy)
    
    # Combine statistical and systematic errors
    err_lo = np.hypot(err_stat_lo, err_sys_lo)
    err_up = np.hypot(err_stat_up, err_sys_up)
    
    return E[mask], y[mask], err_lo[mask], err_up[mask]

def experiment_chi2(model, filename, params, min_energy=10.0, max_energy=1e4):
    """Calculate the chi-squared for a given dataset."""
    xd, yd, err_lo, err_up = load_data(filename, min_energy, max_energy)
    chi2 = 0.
    for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, err_lo, err_up):
        Y_model = model(x_i, params)
        if Y_model > y_i:
            chi2 += np.power((Y_model - y_i) / err_up_i, 2.)
        else:
            chi2 += np.power((Y_model - y_i) / err_lo_i, 2.)
    return chi2

def chi2_function(XG0, delta, Eb, fCO):
    """Calculate total chi-squared across all datasets for both B/C and C/O."""
    chi2 = 0.
    
    # B/C datasets
    bc_files = ['../data/AMS-02_BC_Ekn.txt', 
                '../data/CALET_BC_Ekn.txt', 
                '../data/DAMPE_BC_Ekn.txt']
    
    for filename in bc_files:
        chi2 += experiment_chi2(model_BC_HM, filename, [XG0, delta, Eb, fCO])
    
    # C/O datasets
    co_files = ['../data/CALET_CO_Ekn.txt'] 
                # '../data/AMS-02_CO_R.txt']
    
    for filename in co_files:
        chi2 += experiment_chi2(model_CO_HM, filename, [XG0, delta, Eb, fCO])
    
    return chi2

def fit_BC(initial_params):
    """Perform the chi-squared fit for the B/C model."""
    # Perform minimization
    m = Minuit(chi2_function, XG0=initial_params[0], delta=initial_params[1], Eb=initial_params[2], fCO=initial_params[3])
    m.errordef = Minuit.LEAST_SQUARES

    m.limits["Eb"] = (100., 400.)

    # Optimize using simplex and migrad algorithms
    m.simplex()
    m.migrad()
    m.minos()
    
    # Print fit results
    print(f'X_G  : {m.values[0]:.2f} gr/cm2')
    tau_G = X2tau(m.values[0]) / constants.MYR
    print(f'tau_G  : {tau_G:.2e} Myr')
    print(f'delta : {m.values[1]:.3f}')
    print(f'E_b : {m.values[2]:.3f}')
    print(f'fCO  : {m.values[3]:.3f}')
    print(f'[{m.values[0]:.3f}, {m.values[1]:.3f}, {m.values[2]:.3f}, {m.values[3]:.3f}]')
    dof = 1.  # degrees of freedom placeholder
    return m.values, m.errors, m.fval, dof

if __name__ == "__main__":
    # Initial guess for the parameters
    initial_guess = [4., 0.44, 260., .9]
    fit_BC(initial_guess)
