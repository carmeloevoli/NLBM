import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
from iminuit import Minuit
from dataclasses import dataclass
import math
from jacobi import propagate

# UNITS
year = 3.154e7 # s
Myr = 1e6 * year # s
pc = 3.086e18 # cm
kpc = 1e3 * pc # cm
GeV = 0.00160218 # erg
mbarn = 1e-27 # cm2

# PHYSICAL CONSTANTS
c_light = 3e10 # cm s-1
proton_mass = 1.67262192e-24 # gr

# FIXED PARAMS
sigma_CB = 61. * mbarn # cm^2
sigma_OB = 35. * mbarn # cm^2
sigma_OC = 27. * mbarn # cm^2
n_H = 1. # cm^-3
    
def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    x, y, err_tot_lo, err_tot_up = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
    x = x / norm
    y = norm * np.power(x, slope) * y
    y_err_lo = norm * np.power(x, slope) * err_tot_lo
    y_err_up = norm * np.power(x, slope) * err_tot_up
    ax.errorbar(x, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
                
@dataclass
class GrammageParams:
    XG: float
    Xc: float
    zeta: float
    fCO: float

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

    print(m.params)
    
    E = np.logspace(1, 9, 1000)
    values = GrammageParams(m.values[0], m.values[1], m.values[2], m.values[3])
    y = model_BC(E, values)
    y, ycov = propagate(lambda z: model_BC(E, GrammageParams(z[0], z[1], z[2], z[3])), m.values, m.covariance)

    return E, y, 2. * np.sqrt(np.diag(ycov)), values

def plot_BC(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'B/C')
        ax.set_ylim([0.01, 0.23])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'data/AMS-02_BC.txt', 0., 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/DAMPE_BC.txt', 0., 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/CALET_BC.txt', 0., 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, ycov, values = fit_BC(initialValues)
    ax.plot(E, y, color='tab:blue', zorder=10)
#    ax.fill_between(E, (y - ycov), (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    X_cr = sigma_CB / proton_mass + values.fCO * sigma_OB / proton_mass
    BC_G = X_cr * values.XG
    ax.hlines(BC_G, 1, 1e5, color='tab:blue', linestyle='--', zorder=10)
    ax.text(1.2e3, 0.075, r'$\chi_{\rm G} = 1$ gr/cm$^2$', color='tab:blue', fontsize=22)
    ax.text(20, 0.20, r'$\chi_{\rm 0} = 4.5$ gr/cm$^2$', color='tab:blue', fontsize=22)

    ax.legend(fontsize=20, loc='upper right')
    plt.savefig('NLBM_BC.pdf')

    return values
        
def plot_CO(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'C/O')
        ax.set_ylim([0.65, 1.2])
   
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'data/AMS-02_CO.txt', 0., 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_CO.txt', 0., 1., 'o', 'limegreen', 'CALET', 2)

    E = np.logspace(1, 4, 100)
    y = model_CO(E, initialValues)
    ax.plot(E, y, color='tab:blue', zorder=10)

    #ax.hlines(initialValues.fCO, 1, 1e5, color='tab:blue', linestyle='--', zorder=10)
    #ax.text(20., 0.91, r'$q_C / q_O = 0.9$', color='tab:blue', fontsize=22)
    ax.legend(fontsize=20, loc='upper right')
    plt.savefig('NLBM_CO.pdf')

if __name__== "__main__":
    values = plot_BC(GrammageParams(1., 4.5, 0.1, 0.9))
    plot_CO(values)
    
    print(f'X_G  : {values.XG:5.2f} gr/cm2')
    print(f'X_0  : {values.Xc:5.2f} gr/cm2')
    print(f'zeta : {values.zeta:5.2f}')
    print(f'fCO  : {values.fCO:5.2f}')

    tau = values.XG / (proton_mass * c_light * n_H)
    print(f'tau_G : {tau/Myr}')
