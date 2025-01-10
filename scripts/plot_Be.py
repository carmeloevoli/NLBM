import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

import constants

from fit_BC import model_BeRatio

from utils import set_axes, savefig

def plot_NLBM_Be_ratio(params):
    """Plot Be ratio and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel='$^{10}$Be/$^9$Be', xscale='log', xlim=[0.3, 1e3], ylim=[0., 1.])

    # Plot data for B/C ratio from various sources
    filename = '../data/Be10_Be9_AMS02_TeVPA2024.txt'
    E, r, r_min, r_max = np.loadtxt(filename, usecols=(0,1,2,3), unpack=True)

    color = 'tab:blue'
    ax.errorbar(E, r, yerr=[r - r_min, r_max - r], fmt='o', markeredgecolor=color, color=color, 
                label='AMS-02 (preliminary)', capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=2)

    ax.hlines(0.3, 0.1, 1e4, ls=':', color='tab:gray')

    # Generate and plot the model curve
    E = np.logspace(-1, 3, 1000)
    y = model_BeRatio(E, 40. * constants.MYR, params)
    ax.plot(E, y, color='tab:red', lw=3.3, zorder=10)
    ax.text(29, 0.45, r'$\tau_{\rm G}$ = 40 Myr', color='tab:red', fontsize=20, rotation=50)

    y = model_BeRatio(E, 10. * constants.MYR, params)
    ax.plot(E, y, color='tab:orange', lw=3.3, zorder=10)
    ax.text(7, 0.45, r'$\tau_{\rm G}$ = 10 Myr', color='tab:orange', fontsize=20, rotation=50)
    
    y = model_BeRatio(E, 1. * constants.MYR, params)
    ax.plot(E, y, color='tab:olive', lw=3.3, zorder=10)
    ax.text(0.7, 0.45, r'$\tau_{\rm G}$ = 1 Myr', color='tab:olive', fontsize=20, rotation=50)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='lower right')
    savefig(fig, 'NLBM_Be10_TeVPA2024.pdf')

def estimate_taug(y):
    def grammage_critical(A: float) -> float:
        sigma = 45. * constants.MBARN * np.power(A, 0.67)
        return constants.PROTON_MASS / sigma
    fCO = 0.9
    factor = (constants.SIGMA_O10 + fCO * constants.SIGMA_C10) / (constants.SIGMA_O9 + fCO * constants.SIGMA_C9)
    XG = 1. # gr/cm2
    X10 = grammage_critical(10)
    X9 = grammage_critical(9)
    print (f'critical grammage: {grammage_critical(10.):4.2f}')
    print (f'xsecs factor: {factor:4.2f}')
    print (f'for y: {y} the Be ratio is {factor * (1. + XG / X9) / (1. + XG / X10 + y):4.2f}')

if __name__ == "__main__":
    estimate_taug(2.)
    # Initial guess for the parameters
    params = [1.080, 2.758, 0.184, 0.894]
    plot_NLBM_Be_ratio(params)
