import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math
from dataclasses import dataclass

from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

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
n_H = 1. # 1/cm^3
n_c = 50. # 1/cm^3
E_CR = 1e51 # GeV
mbarn = 1e-27 # cm^2

def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    R, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E = R / norm
    y = norm * np.power(E, slope) * y
    y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)

def get_dsigmadE_ap():
    Eproj_table, Epbar_table, pp_table, pHe_table, Hep_table, HeHe_table = np.loadtxt('data/supplementary__XS_table_Param_II_B.txt', usecols=(0,1,2,3,8,9), unpack=True)
    projectile_kinetic_energy = np.linspace(np.log10(1.), np.log10(1e7), 211)
    secondary_kinetic_energy = np.linspace(np.log10(0.1), np.log10(1e4), 151)
    points = (secondary_kinetic_energy, projectile_kinetic_energy)

    pp_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    pHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    Hep_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    HeHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    count = 0
    for j in range(len(projectile_kinetic_energy)):
        pp_xsec[:,j] = pp_table[count:count + 151] * 1e31 # m2 -> mbarn
        pHe_xsec[:,j] = pHe_table[count:count + 151] * 1e31 # m2 -> mbarn
        Hep_xsec[:,j] = Hep_table[count:count + 151] * 1e31 # m2 -> mbarn
        HeHe_xsec[:,j] = HeHe_table[count:count + 151] * 1e31 # m2 -> mbarn
        count = count + 151
    pp_i = RegularGridInterpolator(points, pp_xsec, bounds_error=False, fill_value=0.)
    pHe_i = RegularGridInterpolator(points, pHe_xsec, bounds_error=False, fill_value=0.)
    Hep_i = RegularGridInterpolator(points, Hep_xsec, bounds_error=False, fill_value=0.)
    HeHe_i = RegularGridInterpolator(points, HeHe_xsec, bounds_error=False, fill_value=0.)

    return pp_i, pHe_i, Hep_i, HeHe_i

def gest_dsigmadE_pos():
    Eproj_table, Epbar_table, pp_table, pHe_table, Hep_table, HeHe_table = np.loadtxt('data/supplementary_table_positrons_best_fit.dat', usecols=(0,1,2,3,8,9), unpack=True)

    projectile_kinetic_energy = np.linspace(np.log10(0.1), np.log10(1e6), 140)
    secondary_kinetic_energy = np.linspace(np.log10(0.01), np.log10(1e4), 90)
    points = (secondary_kinetic_energy, projectile_kinetic_energy)

    pp_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    pHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    Hep_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    HeHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    count = 0
    for j in range(len(projectile_kinetic_energy)):
        pp_xsec[:,j] = pp_table[count:count + 90] # mbarn
        pHe_xsec[:,j] = pHe_table[count:count + 90] # mbarn
        Hep_xsec[:,j] = Hep_table[count:count + 90] # mbarn
        HeHe_xsec[:,j] = HeHe_table[count:count + 90] # mbarn
        count = count + 90
    pp_i = RegularGridInterpolator(points, pp_xsec, bounds_error=False, fill_value=0.)
    pHe_i = RegularGridInterpolator(points, pHe_xsec, bounds_error=False, fill_value=0.)
    Hep_i = RegularGridInterpolator(points, Hep_xsec, bounds_error=False, fill_value=0.)
    HeHe_i = RegularGridInterpolator(points, HeHe_xsec, bounds_error=False, fill_value=0.)

    return pp_i, pHe_i, Hep_i, HeHe_i

@dataclass
class InjectionParams:
    A: float
    lnEb: float
    p: float
    dp: float
    s: float
    tauG: float

@dataclass
class GrammageParams:
    tauG: float
    tau0: float
    zeta: float

def primaryModel(E, par):
    value = snrate * par.tauG / V_G # cm^-3
    value *= E_CR / np.power(E_0, 2.) * (par.p - 2.) # GeV^-1 cm^-3
    value *= cLight / 4. / math.pi # GeV^-1 cm^-2 s^-1 sr^-1
    Eb = np.exp(par.lnEb)
    y = par.A * value * (E / E_0)**(-par.p)
    y *= np.power(1. + np.power(E / Eb, par.dp / par.s), par.s)
    return y

def I_ap():
    Ip_par = InjectionParams(14.864, np.log(639.935), 2.811, 0.251, 0.123, 0.937 * Myr)
    Ihe_par = InjectionParams(0.948, np.log(474.829), 2.737, 0.308, 0.192, 0.937 * Myr)
        
    chi_par = GrammageParams(0.937 * Myr, 0.079 * Myr, 0.080)
    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = get_dsigmadE_ap()

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        return Eprime * chi_par.tauG * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        tau_c = chi_par.tau0 * np.power(Eprime, -chi_par.zeta * np.log(Eprime))
        return Eprime * tau_c * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)

    size = 100
    E = np.logspace(0, 4, size)
    value_G = np.zeros(size)
    for i in range(size):
        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
    value_G *= cLight * n_H
    
    value_c = np.zeros(size)
    for i in range(size):
        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
    value_c *= cLight * n_c

    return E, value_c, value_G

def I_pos():
    Ip_par = InjectionParams(14.864, np.log(639.935), 2.811, 0.251, 0.123, 0.937 * Myr)
    Ihe_par = InjectionParams(0.948, np.log(474.829), 2.737, 0.308, 0.192, 0.937 * Myr)
        
    chi_par = GrammageParams(0.937 * Myr, 0.079 * Myr, 0.080)
    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = gest_dsigmadE_pos()

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        return Eprime * chi_par.tauG * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
        tau_c = chi_par.tau0 * np.power(Eprime, -chi_par.zeta * np.log(Eprime))
        return Eprime * tau_c * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)

    size = 100
    E = np.logspace(0, 4, size)
    value_G = np.zeros(size)
    for i in range(size):
        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
    value_G *= cLight * n_H
    
    value_c = np.zeros(size)
    for i in range(size):
        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
    value_c *= cLight * n_c

    return E, value_c, value_G
    
def plot_antiprotons():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([0.5, 10.])
        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'kiss_tables/AMS-02_pbar_rigidity.txt', 2.7, 1., 'o', 'red', r'$\bar p$', 1)
    plot_data(ax, 'kiss_tables/AMS-02_e+_rigidity.txt', 2.7, 1., 'o', 'blue', r'$e^+$', 1)
    
    f_He = 1.2
    
    E, I_c, I_G = I_ap()
    ax.plot(E, np.power(E, 2.7) * I_G * f_He * 1e4, 'darkorange', linestyle='--') # cm -> m
    ax.plot(E, np.power(E, 2.7) * (I_G + I_c) * f_He * 1e4, 'darkorange', linestyle='-')

    E, I_c, I_G = I_pos()
    ax.plot(E, np.power(E, 2.7) * I_G * f_He * 1e4, 'tab:cyan', linestyle='--') # cm -> m
    ax.plot(E, np.power(E, 2.7) * (I_G + I_c) * f_He * 1e4, 'tab:cyan', linestyle='-')

    ax.legend(fontsize=30, loc='best')
    plt.savefig('NLBM_ap.pdf')

def plot_antiprotons_ratio():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        #ax.set_xscale('log')
        ax.set_xlim([1e1, 4e2])
        ax.set_ylabel(r'ratio')
        ax.set_ylim([0.2, 0.8])
        #ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'kiss_tables/AMS-02_pbar_e+_rigidity.txt', 0., 1., 'o', 'red', r'$\bar p$/$e^+$', 1)
    
    f_He = 1.2

    E, I_c, I_G = I_ap()
    I_ap_flux = I_c + I_G
    E, I_c, I_G = I_pos()
    I_pos_flux = I_c + I_G
    
    ax.plot(E, I_ap_flux / I_pos_flux, 'darkorange', linestyle='-')

    ax.legend(fontsize=30, loc='best')
    plt.savefig('NLBM_ap_ratio.pdf')
    
if __name__== "__main__":
    plot_antiprotons()
    plot_antiprotons_ratio()
