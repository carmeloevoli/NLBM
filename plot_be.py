import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

# FIXED PARAMS

year = 3.1e7 # s
Myr = 1e6 * year # s
mp = 0.938 # GeV
sigma_C10 = 3.8
sigma_C9 = 5.7
sigma_O10 = 2.4
sigma_O9 = 3.9
f_OC = 0.9
    
def plot_data(ax, filename, zorder=1):
    label = 'AMS-02 2023 (preliminary)'
    color = 'crimson'
    E, y_min, y_mean, y_max = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
    y_err_lo = y_mean - y_min
    y_err_up = y_max - y_mean
    ax.errorbar(E, y_mean, yerr=[y_err_lo, y_err_up], fmt='o', markeredgecolor=color, color=color,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)

def compute_be(tau_G):
    E = np.linspace(0, 12, 1000)
    gamma_Be = (E + mp) / mp
    tau_d = 2. * Myr * gamma_Be
    sigma_10 = sigma_C10 + f_OC * sigma_O10
    sigma_9 = sigma_C9 + f_OC * sigma_O9
    value = sigma_10 / sigma_9 * tau_d / (tau_G + tau_d)
    return E, value

def plot_be():
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xlim([0.01, 12])
        ax.set_ylabel(r'Be$^{10}$ / Be$^9$')
        ax.set_ylim([0., 0.75])
#        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'data/AMS-02_Be_ratio_preliminary.txt')

    E, r = compute_be(0.5 * Myr)
    ax.plot(E, r, color='tab:blue', label='0.5 Myr')

    E, r = compute_be(5 * Myr)
    ax.plot(E, r, color='tab:orange', label='5 Myr')

    E, r = compute_be(20 * Myr)
    ax.plot(E, r, color='tab:green', label='20 Myr')

    sigma_10 = sigma_C10 + f_OC * sigma_O10
    sigma_9 = sigma_C9 + f_OC * sigma_O9

    ax.hlines(sigma_10 / sigma_9, 0, 20, ls='--', color='tab:blue')
    ax.text(8., 0.67, r'$\sigma_{10}/\sigma_9$', color='tab:blue')
    
    ax.text(1, 0.05, 'AMS-02 (2023) preliminary', fontsize=20)
    
    ax.legend(fontsize=18, loc='lower right')
    plt.savefig('NLBM_Be_ratio.pdf')
  
if __name__== "__main__":
    plot_be()
