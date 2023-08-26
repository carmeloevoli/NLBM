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
tau_G = 2. * Myr # s
hdisk = 100. * pc
R_G = 10. * kpc
V_G = math.pi * np.power(R_G, 2.) * 2. * hdisk
n_s = 2000. / V_G
q_0 = 11.3e43 # TBD
beta = 2.80

def I_H():
    E = np.logspace(1, 4, 1000)
    value = cLight / 4. / math.pi
    value *= n_s * tau_G
    value *= q_0 * np.power(E, -beta)
    return E, value
    
def B_C():
    E = np.logspace(1, 4, 1000)
    T = E / E[0]
    value = 0.15 * np.power(T, -0.1 * np.log(T))
    value += 0.02
    return E, value
    
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
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'B/C')
        ax.set_yscale('log')
        #ax.set_ylim(ylim)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'kiss_tables/AMS-02_B_C_kineticEnergyPerNucleon.txt', 0.33, 1., 'o', 'tab:blue', 'AMS-02')
    plot_data(ax, 'kiss_tables/CALET_B_C_kineticEnergyPerNucleon.txt', 0.33, 1., 'o', 'tab:orange', 'CALET', 3)
    plot_data(ax, 'kiss_tables/DAMPE_B_C_kineticEnergyPerNucleon.txt', 0.33, 1., 'o', 'tab:red', 'DAMPE', 3)

    E, bc = B_C()
    ax.plot(E, np.power(E, 0.3) * bc, 'tab:gray')

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

    plot_data(ax, 'kiss_tables/AMS-02_H_rigidity.txt', 2.7, 1., 'o', 'tab:blue', 'AMS-02')
    plot_data(ax, 'kiss_tables/CALET_H_kineticEnergy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 3)
    plot_data(ax, 'kiss_tables/DAMPE_H_totalEnergy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

    E, I = I_H()
    ax.plot(E, np.power(E, 2.7) * I, 'tab:gray')

    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('NLBM_H.pdf')
    
if __name__== "__main__":
    plot_BC()
    plot_protons()
