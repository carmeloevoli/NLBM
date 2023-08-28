import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math

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

def I_H():
    E = np.logspace(0, 4, 1000)
    value = cLight / 4. / math.pi
    value *= snrate * tau_G / V_G
    value *= E_CR / np.power(E_0, 2.) * (p - 2.) * np.power(E / E_0, -p)
    return E, value
    
def B_C():
    E = np.logspace(0, 4, 1000)
    value_G = cLight * sigmaCB * tau_G * n_H * (E / E)
    value_c = cLight * sigmaCB * n_c * tau_0 * np.power(E, -zeta * np.log(E))
    return E, value_c, value_G
    
def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E *= norm
    y = np.power(E, slope) * y
    y_err_lo = np.power(E, slope) * err_stat_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = np.power(E, slope) * err_stat_up # np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
    y_err_lo = np.power(E, slope) * err_sys_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = np.power(E, slope) * err_sys_lo # np.sqrt(err_stat_up**2. + err_sys_up**2.)
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
    
if __name__== "__main__":
    plot_BC()
    plot_protons()
