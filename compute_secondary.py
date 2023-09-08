import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math

def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    R, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E = R / norm
    y = norm * np.power(E, slope) * y
    y_err_lo = norm * np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
    
def plot_antiprotons():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{2.7}$ I')
        ax.set_ylim([0., 7.])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'kiss_tables/AMS-02_pbar_rigidity.txt', 2.7, 1., 'o', 'darkorange', r'$\bar p$', 1)
    plot_data(ax, 'kiss_tables/AMS-02_e+_rigidity.txt', 2.7, 1., 'o', 'red', r'$e^+$', 1)


#    plot_data(ax, 'kiss_tables/AMS-02_H_rigidity.txt', 2.7, 1.8e-4, 'o', 'lightgray', 'protons [1.8 x 10$^{-4}$]')
#    plot_data(ax, 'kiss_tables/CALET_H_kineticEnergy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 3)
#    plot_data(ax, 'kiss_tables/DAMPE_H_totalEnergy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

#    E, I_c, I_G = I_ap()
#    I_G *= 1.9
#    I_c *= 5.6
#    ax.plot(E, np.power(E, 2.7) * I_G, 'tab:gray', linestyle='--')
#    ax.plot(E, np.power(E, 2.7) * I_c, 'tab:gray', linestyle=':')
#    ax.plot(E, np.power(E, 2.7) * (I_G + I_c), 'tab:gray', linestyle='-')

    ax.legend(fontsize=30, loc='upper right')
    plt.savefig('NLBM_ap.pdf')

if __name__== "__main__":
    plot_antiprotons()
    #plot_positrons()
