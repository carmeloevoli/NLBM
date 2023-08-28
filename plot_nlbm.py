import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math

from scipy.integrate import quad
from scipy.interpolate import interpn

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

# Free parameters
tau_G = 1. * Myr # s
tau_0 = 0.09 * Myr # s
E_CR = 1.4e56 # GeV
p = 2.80
xi = 1.

# Tables
def dsigmadE_table():
    Eproj_table, Epbar_table, dsigmadE_table = np.loadtxt('data/supplementary__XS_table_Param_II_B.txt', usecols=(0,1,2), unpack=True)
    projectile_kinetic_energy = np.logspace(np.log10(1.), np.log10(1e7), 211)
    secondary_kinetic_energy = np.logspace(np.log10(0.1), np.log10(1e4), 151)
    cross_section = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    count = 0
    for j in range(len(projectile_kinetic_energy)):
        cross_section[:,j] = dsigmadE_table[count:count + 151] * 1e4 # m2 -> cm2
        count = count + 151
    points = (secondary_kinetic_energy, projectile_kinetic_energy)
    return points, cross_section

def I_H():
    E = np.logspace(0, 4, 1000)
    value = cLight / 4. / math.pi
    value *= snrate * tau_G / V_G
    value *= E_CR / np.power(E_0, 2.) * (p - 2.) * np.power(E / E_0, -p)
    return E, value

def I_ap():
    axis, table = dsigmadE_table()
    
    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        dsigmadE = interpn(axis, table, (E, Eprime), bounds_error=False, fill_value=0.)
        return Eprime * dsigmadE * np.power(Eprime / E_0, -p)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        dsigmadE = interpn(axis, table, (E, Eprime), bounds_error=False, fill_value=0.)
        tau_c = tau_0 * np.power(Eprime, -zeta * np.log(Eprime))
        return Eprime * dsigmadE * tau_c * np.power(Eprime / E_0, -p)

    size = 100
    E = np.logspace(0, 4, size)
    value_G = np.zeros(size)
    for i in range(size):
        integral = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))
        value_G[i] = snrate * tau_G / V_G * E_CR / np.power(E_0, 2.) * (p - 2.) * integral[0]
    value_G *= cLight * n_H * xi * tau_G
    
    value_c = np.zeros(size)
    for i in range(size):
        integral = quad(integrand_qc, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))
        value_c[i] = snrate / V_G * E_CR / np.power(E_0, 2.) * (p - 2.) * integral[0]
    value_c *= cLight * n_c * xi * tau_G
    
    return E, cLight / 4. / math.pi * value_c, cLight / 4. / math.pi * value_G
    
def B_C():
    E = np.logspace(0, 4, 1000)
    value_G = cLight * sigmaCB * tau_G * n_H * (E / E)
    value_c = cLight * sigmaCB * n_c * tau_0 * np.power(E, -zeta * np.log(E))
    return E, value_c, value_G

def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    y = norm * np.power(E, slope) * y
    y_err_lo = norm * np.power(E, slope) * err_stat_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * err_stat_up # np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
    y_err_lo = norm * np.power(E, slope) * err_sys_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * err_sys_lo # np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
                
def plot_BC():
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e0, 1e4])
        ax.set_ylabel(r'B/C')
        ax.set_yscale('log')
        #ax.set_ylim(ylim)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'kiss_tables/AMS-02_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'tab:olive', 'AMS-02')
    plot_data(ax, 'kiss_tables/CALET_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'tab:orange', 'CALET', 3)
    plot_data(ax, 'kiss_tables/DAMPE_B_C_kineticEnergyPerNucleon.txt', 0., 1., 'o', 'tab:red', 'DAMPE', 3)

    E, bc_c, bc_g = B_C()
    ax.plot(E, bc_c + bc_g, 'tab:gray')
    ax.plot(E, bc_g, 'tab:gray', linestyle=':')

    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('NLBM_BC.pdf')

def plot_protons():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.7}$ I')
        #ax.set_yscale('log')
        ax.set_ylim([6e3, 16e3])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'kiss_tables/AMS-02_H_rigidity.txt', 2.7, 1., 'o', 'tab:olive', 'AMS-02')
    plot_data(ax, 'kiss_tables/CALET_H_kineticEnergy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 3)
    plot_data(ax, 'kiss_tables/DAMPE_H_totalEnergy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

    E, I = I_H()
    ax.plot(E, np.power(E, 2.7) * I, 'tab:gray')

    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('NLBM_H.pdf')
    
def plot_antiprotons():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{2.7}$ I')
        #ax.set_yscale('log')
        #ax.set_ylim([6e3, 16e3])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'kiss_tables/AMS-02_H_rigidity.txt', 2.7, 1.8e-4, 'o', 'lightgray', 'protons [1.8 x 10$^{-4}$]')
    plot_data(ax, 'kiss_tables/AMS-02_pbar_rigidity.txt', 2.7, 1., 'o', 'tab:orange', 'AMS-02', 3)
#    plot_data(ax, 'kiss_tables/CALET_H_kineticEnergy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 3)
#    plot_data(ax, 'kiss_tables/DAMPE_H_totalEnergy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

    E, I_c, I_G = I_ap()
    I_G *= 1.9
    I_c *= 5.6
    ax.plot(E, np.power(E, 2.7) * I_G, 'tab:gray', linestyle='--')
    ax.plot(E, np.power(E, 2.7) * I_c, 'tab:gray', linestyle=':')
    ax.plot(E, np.power(E, 2.7) * (I_G + I_c), 'tab:gray', linestyle='-')

    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('NLBM_ap.pdf')

def test_apxsec():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([0.1, 1e4])
        #ax.set_ylabel(r'E$^{2.7}$ I')
        ax.set_yscale('log')
        ax.set_ylim([1e-35, 1e-31])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    axis, table = dsigmadE_table()
    
    Eap = np.logspace(0, 4, 1000)
    s = np.zeros(1000)
    for i in range(1000):
        s[i] = interpn(axis, table, (Eap[i], 450.))
    
    ax.plot(Eap, s)
    
    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('ap_xecs.pdf')

if __name__== "__main__":
#    test_apxsec()
#    plot_BC()
#    plot_protons()
    plot_antiprotons()
