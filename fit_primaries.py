import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
from iminuit import Minuit
from dataclasses import dataclass
import math
from jacobi import propagate
from fit_model import plot_data

# UNITS
year = 3.154e7 # s
Myr = 1e6 * year # s
pc = 3.086e18 # cm
kpc = 1e3 * pc # cm
erg2GeV = 624.151

# PHYSICAL CONSTANTS
c_light = 3e10 # cm s-1

# FIXED PARAMS
sn_rate = 1. / 50 / year # s-1
h_disk = 100. * pc # cm
R_G = 10. * kpc # cm
V_G = math.pi * np.power(R_G, 2.) * 2. * h_disk # cm3
E_0 = 10. # GeV
E_CR = 1e51 * erg2GeV # GeV

@dataclass
class InjectionParams:
    xi: float
    lnEb: float
    alpha: float
    dalpha: float
    s: float
    tauG: float

def model_primary(E, par, doBreak = True):
    value = sn_rate * par.tauG / V_G # cm^-3
    value *= par.xi * E_CR / np.power(E_0, 2.) * (par.alpha - 2.) # GeV^-1 cm^-3
    value *= c_light / 4. / math.pi # GeV^-1 cm^-2 s^-1 sr^-1
    Eb = np.exp(par.lnEb)
    y = value * (E / E_0)**(-par.alpha)
    if doBreak:
        y *= np.power(1. + np.power(E / Eb, par.dalpha / par.s), par.s)
    return y * 1e4 # GeV^-1 m^-2 s^-1 sr^-1

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

    print(m.params)

    E = np.logspace(1, 9, 1000)
    values = InjectionParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5])
    y = model_primary(E, values)
    y, ycov = propagate(lambda z: model_primary(E, InjectionParams(z[0], z[1], z[2], z[3], z[4], z[5])), m.values, m.covariance)

    return E, y, 2. * np.sqrt(np.diag(ycov)), values
    
def plot_H(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([6e3, 18e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    values = []
    
    plot_data(ax, 'data/AMS-02_H.txt', 2.7, 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_H.txt', 2.7, 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/DAMPE_H.txt', 2.7, 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, ycov, values = fit_H(initialValues)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:blue', zorder=10)
    
    y = model_primary(E, values, False)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:blue', linestyle='--', zorder=10)
    
#    ax.fill_between(E, np.power(E, 2.7) * 1e4 * (y - ycov), np.power(E, 2.7) * 1e4 * (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    ax.legend(fontsize=20, loc='upper left')
    plt.savefig('NLBM_H.pdf')
    
    return values

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

    print(m.params)

    E = np.logspace(1, 9, 1000)
    values = InjectionParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5])
    y = model_primary(E, values)
    y, ycov = propagate(lambda z: model_primary(E, InjectionParams(z[0], z[1], z[2], z[3], z[4], z[5])), m.values, m.covariance)

    return E, y, 2. * np.sqrt(np.diag(ycov)), values

def plot_He(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV/n$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([0.2e3, 2e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    values = []
    
    plot_data(ax, 'data/AMS-02_He.txt', 2.7, 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_He.txt', 2.7, 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/DAMPE_He.txt', 2.7, 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, ycov, values = fit_He(initialValues)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:blue', zorder=10)
##    ax.fill_between(E, np.power(E, 2.7) * 1e4 * (y - ycov), np.power(E, 2.7) * 1e4 * (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    y = model_primary(E, values, False)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:blue', linestyle='--', zorder=10)

    ax.legend(fontsize=20, loc='upper left')
    plt.savefig('NLBM_He.pdf')

    return values

if __name__== "__main__":
    hPar = plot_H(InjectionParams(0.05, np.log(1e3), 2.85, 0.25, 0.1, 0.5 * Myr))
    hePar = plot_He(InjectionParams(0.05, np.log(1e3), 2.85, 0.25, 0.1, 0.5 * Myr))

    print(f'H_xi = {hPar.xi:5.3f}')
    print(f'H_Eb = {np.exp(hPar.lnEb):5.3f} GeV')
    print(f'H_alpha = {hPar.alpha:5.3f}')
    print(f'H_daplha = {hPar.dalpha:5.3f}')
    print(f'H_s = {hPar.s:5.3f}')

    print(f'He_xi = {hePar.xi:5.3f}')
    print(f'He_Eb = {np.exp(hePar.lnEb):5.3f} GeV')
    print(f'He_alpha = {hePar.alpha:5.3f}')
    print(f'He_daplha = {hePar.dalpha:5.3f}')
    print(f'He_s = {hePar.s:5.3f}')
