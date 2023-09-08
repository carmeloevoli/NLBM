import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
from iminuit import Minuit
from dataclasses import dataclass
import math
from jacobi import propagate

# FIXED PARAMS

year = 3.1e7 # s
Myr = 1e6 * year # s
pc = 3.1e18 # cm
kpc = 1e3 * pc # cm
cLight = 3e10 # cm / s
snrate = 1. / 50 / year # 1/s
hdisk = 100. * pc # cm
R_G = 10. * kpc # cm
V_G = math.pi * np.power(R_G, 2.) * 2. * hdisk # cm^3
E_0 = 10. # GeV
sigmaCB = 60e-27 # cm^2
zeta = 0.1
n_H = 1. # 1/cm^3
n_c = 50. # 1/cm^3
E_CR = 1e55 # GeV

tau_G = 29.0e12 # s

def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    R, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E = R / norm
    y = norm * np.power(E, slope) * y
    y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)

@dataclass
class ModelParams:
    A: float
    lnEb: float
    p: float
    dp: float
    s: float

def model(E, par):
    value = cLight / 4. / math.pi
    value *= snrate * tau_G / V_G
    value *= E_CR / np.power(E_0, 2.) * (par.p - 2.)
    Eb = np.exp(par.lnEb)
    y = par.A * (E / E_0)**(-par.p)
    y *= np.power(1. + np.power(E / Eb, par.dp / par.s), par.s)
    return y

def fit_H(par):
    def get_data(filename, norm, minEnergy, maxEnergy = 1e20):
        slope = 0.
        E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
        E = E / norm
        y = norm * np.power(E, slope) * y
        y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
        y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
        items = [i for i in range(len(E)) if (E[i] > minEnergy and E[i] < maxEnergy)]
        return E[items], y[items], y_err_lo[items], y_err_up[items]
  
    def experiment_chi2(filename, par, norm = 1):
        xd, yd, errd_lo, errd_up = get_data(filename, norm, 30., 1e4)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, errd_lo, errd_up):
            Y = model(x_i, par)
            if Y > y_i:
                chi2 += np.power((Y - y_i) / err_up_i, 2.)
            else:
                chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2
 
    def chi2_function(A, lnEb, p, dp, s):
        chi2 = 0.
        chi2 += experiment_chi2('kiss_tables/AMS-02_H_rigidity.txt', ModelParams(A, lnEb, p, dp, s), 1.)
        chi2 += experiment_chi2('kiss_tables/DAMPE_H_totalEnergy.txt', ModelParams(A, lnEb, p, dp, s), 1.)
        chi2 += experiment_chi2('kiss_tables/CALET_H_kineticEnergy.txt', ModelParams(A, lnEb, p, dp, s), 1.)
        return chi2

    m = Minuit(chi2_function, A=par.A, lnEb=par.lnEb, p=par.p, dp=par.dp, s=par.s)
    m.errordef = Minuit.LEAST_SQUARES
    #m.fixed['s'] = True
    m.limits['s'] = (0.01, 1.)
    
    m.simplex()
    m.migrad()
    m.minos()

    print(m.params)

    E = np.logspace(1, 9, 1000)
    y = model(E, ModelParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4]))
    y, ycov = propagate(lambda z: model(E, ModelParams(z[0], z[1], z[2], z[3], z[4])), m.values, m.covariance)

    return E, y, np.sqrt(np.diag(ycov))
    
def plot_H():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([6e3, 16e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    filename = 'kiss_tables/AMS-02_H_rigidity.txt'
    plot_data(ax, filename, 2.7, 1., 'o', 'darkorange', 'AMS-02', 1)

    filename = 'kiss_tables/DAMPE_H_totalEnergy.txt'
    plot_data(ax, filename, 2.7, 1., 'o', 'limegreen', 'DAMPE', 2)

    filename = 'kiss_tables/CALET_H_kineticEnergy.txt'
    plot_data(ax, filename, 2.7, 1., 'o', 'royalblue', 'CALET', 3)

    E, y, ycov = fit_H(ModelParams(30., np.log(1e3), 2.85, 0.25, 0.1))
    ax.plot(E, np.power(E, 2.7) * y, color='tab:purple', zorder=10)
    ax.fill_between(E, np.power(E, 2.7) * (y - ycov), np.power(E, 2.7) * (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    ax.legend(fontsize=18, loc='upper left')
    plt.savefig('NLBM_H.pdf')

def fit_He(par):
    def get_data(filename, norm, minEnergy, maxEnergy = 1e20):
        slope = 0.
        E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
        E = E / norm
        y = norm * np.power(E, slope) * y
        y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
        y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
        items = [i for i in range(len(E)) if (E[i] > minEnergy and E[i] < maxEnergy)]
        return E[items], y[items], y_err_lo[items], y_err_up[items]
  
    def experiment_chi2(filename, par, norm = 1):
        xd, yd, errd_lo, errd_up = get_data(filename, norm, 30., 1e4)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, errd_lo, errd_up):
            Y = model(x_i, par)
            if Y > y_i:
                chi2 += np.power((Y - y_i) / err_up_i, 2.)
            else:
                chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2
 
    def chi2_function(A, lnEb, p, dp, s):
        chi2 = 0.
        chi2 += experiment_chi2('kiss_tables/AMS-02_He_rigidity.txt', ModelParams(A, lnEb, p, dp, s), 2.)
        chi2 += experiment_chi2('kiss_tables/DAMPE_He_totalEnergy.txt', ModelParams(A, lnEb, p, dp, s), 4.)
        chi2 += experiment_chi2('kiss_tables/CALET_He_kineticEnergy.txt', ModelParams(A, lnEb, p, dp, s), 4.)
        return chi2

    m = Minuit(chi2_function, A=par.A, lnEb=par.lnEb, p=par.p, dp=par.dp, s=par.s)
    m.errordef = Minuit.LEAST_SQUARES
    #m.fixed['s'] = True
    m.limits['s'] = (0.01, 1.)

    m.simplex()
    m.migrad()
    m.minos()

    print(m.params)

    E = np.logspace(1, 9, 1000)
    y = model(E, ModelParams(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4]))
    y, ycov = propagate(lambda z: model(E, ModelParams(z[0], z[1], z[2], z[3], z[4])), m.values, m.covariance)

    return E, y, np.sqrt(np.diag(ycov))

def plot_He():
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
    
    filename = 'kiss_tables/AMS-02_He_rigidity.txt'
    plot_data(ax, filename, 2.7, 2., 'o', 'darkorange', 'AMS-02', 1)
    
    filename = 'kiss_tables/DAMPE_He_totalEnergy.txt'
    plot_data(ax, filename, 2.7, 4., 'o', 'limegreen', 'DAMPE', 2)

    filename = 'kiss_tables/CALET_He_kineticEnergy.txt'
    plot_data(ax, filename, 2.7, 4., 'o', 'royalblue', 'CALET', 3)

    E, y, ycov = fit_He(ModelParams(30., np.log(1e3), 2.85, 0.25, 0.1))
    ax.plot(E, np.power(E, 2.7) * y, color='tab:purple', zorder=10)
    ax.fill_between(E, np.power(E, 2.7) * (y - ycov), np.power(E, 2.7) * (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    ax.legend(fontsize=18, loc='upper left')
    plt.savefig('NLBM_He.pdf')

@dataclass
class GrammageParams:
    tauG: float
    tau0: float
    zeta: float

def grammage(E, par):
    value_G = cLight * sigmaCB * par.tauG * n_H
    value_c = cLight * sigmaCB * n_c * par.tau0 * np.power(E, -par.zeta * np.log(E))
    return value_c + value_G

def fit_BC(par):
    def get_data(filename, norm, minEnergy, maxEnergy = 1e20):
        slope = 0.
        E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
        E = E / norm
        y = norm * np.power(E, slope) * y
        y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
        y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
        items = [i for i in range(len(E)) if (E[i] > minEnergy and E[i] < maxEnergy)]
        return E[items], y[items], y_err_lo[items], y_err_up[items]
  
    def experiment_chi2(filename, par, norm = 1):
        xd, yd, errd_lo, errd_up = get_data(filename, norm, 30.)
        chi2 = 0.
        for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, errd_lo, errd_up):
            Y = grammage(x_i, par)
            if Y > y_i:
                chi2 += np.power((Y - y_i) / err_up_i, 2.)
            else:
                chi2 += np.power((Y - y_i) / err_lo_i, 2.)
        return chi2
 
    def chi2_function(tauG, tau0, zeta):
        chi2 = 0.
        chi2 += experiment_chi2('kiss_tables/AMS-02_B_C_kineticEnergyPerNucleon.txt', GrammageParams(tauG, tau0, zeta), 1.)
        chi2 += experiment_chi2('kiss_tables/CALET_B_C_kineticEnergyPerNucleon.txt', GrammageParams(tauG, tau0, zeta), 1.)
        chi2 += experiment_chi2('kiss_tables/DAMPE_B_C_kineticEnergyPerNucleon.txt', GrammageParams(tauG, tau0, zeta), 1.)
        return chi2

    m = Minuit(chi2_function, tauG=par.tauG, tau0=par.tau0, zeta=par.zeta)
    m.errordef = Minuit.LEAST_SQUARES
    #m.fixed['s'] = True
    #m.limits['s'] = (0.01, 1.)

    m.simplex()
    m.migrad()
    m.minos()

    print(m.params)

    E = np.logspace(1, 9, 1000)
    y = grammage(E, GrammageParams(m.values[0], m.values[1], m.values[2]))
    y, ycov = propagate(lambda z: grammage(E, GrammageParams(z[0], z[1], z[2])), m.values, m.covariance)

    return E, y, np.sqrt(np.diag(ycov))
    
def plot_BC():
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'B/C')
        #ax.set_yscale('log')
        ax.set_ylim([0, 0.25])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'kiss_tables/AMS-02_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'darkorange', 'AMS-02', 1)
    plot_data(ax, 'kiss_tables/CALET_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'limegreen', 'DAMPE', 2)
    plot_data(ax, 'kiss_tables/DAMPE_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'royalblue', 'CALET', 3)

    E, y, ycov = fit_BC(GrammageParams(Myr, 0.1 * Myr, 0.1))
    ax.plot(E, y, color='tab:purple', zorder=10)
    ax.fill_between(E, (y - ycov), (y + ycov), color='tab:purple', alpha=0.25, zorder=10)

    ax.legend(fontsize=18, loc='lower left')
    plt.savefig('NLBM_BC.pdf')

if __name__== "__main__":
    plot_BC()
    plot_H()
    plot_He()
