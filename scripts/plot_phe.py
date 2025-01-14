import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig
from fit_phe import model_primary

def plot_H(params):
    """Plot proton spectrum data and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ylabel=r'$E^{2.7}$ $I_{\rm H}$ [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'
    set_axes(ax, xlabel='E [GeV]', ylabel=ylabel, xscale='log', xlim=[3e1, 1e4], ylim=[7e3, 16e3])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    # Plot data for proton spectrum from various sources
    plot_data(ax, 'AMS-02_H_Ek.txt', 2.7, 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'CALET_H_Ek.txt', 2.7, 1., 'o', 'tab:green', 'CALET', 2)
    plot_data(ax, 'DAMPE_H_Ek.txt', 2.7, 1., 'o', 'tab:brown', 'DAMPE', 3)

    # Generate and plot the model curve
    E = np.logspace(1, 4, 1000) 
    y = model_primary(E, params)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:red', lw=3, zorder=10)

    # Add text annotations
    ax.text(38., 9.6e3, r'$\alpha = 2.78$', color='tab:red', fontsize=22)
    ax.text(200., 8e3, r'$E_{\rm b} = 550$ GeV', color='tab:red', fontsize=22)
    ax.text(3e3, 9.6e3, r'$\Delta \alpha = 0.2$', color='tab:red', fontsize=22)

    ax.fill_between([1., 50.], 1e3, 1e5, color='tab:gray', alpha=0.20, zorder=1)

    # Show legend and save figure
    ax.legend(fontsize=20, loc='best')
    savefig(fig, 'NLBM_H.pdf')

def plot_He(params):
    """Plot Helium spectrum data and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ylabel=r'$E^{2.7}$ $I_{\rm He}$ [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'
    set_axes(ax, xlabel='E [GeV/n]', ylabel=ylabel, xscale='log', xlim=[3e1, 1e4], ylim=[0.2e3, 2e3])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,2))

    # Plot data for B/C ratio from various sources
    plot_data(ax, 'AMS-02_He_Ekn.txt', 2.7, 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'DAMPE_He_Ekn.txt', 2.7, 1., 'o', 'tab:brown', 'CALET', 3)
    plot_data(ax, 'CALET_He_Ekn.txt', 2.7, 1., 'o', 'tab:green', 'DAMPE', 2)

    # Generate and plot the model curve
    E = np.logspace(1, 4, 1000)
    y = model_primary(E, params)
    ax.plot(E, np.power(E, 2.7) * y, color='tab:red', lw=3, zorder=10)

    # # Add text annotations
    ax.text(40., 0.91e3, r'$\alpha = 2.68$', color='tab:red', fontsize=22)
    ax.text(220., 4.6e2, r'$E_{\rm b} = 370$ GeV', color='tab:red', fontsize=22)
    ax.text(3.6e3, 8.9e2, r'$\Delta \alpha = 0.2$', color='tab:red', fontsize=22)

    ax.fill_between([1., 50.], 1e2, 1e4, color='tab:gray', alpha=0.20, zorder=1)

    # # Show legend and save figure
    # ax.legend(fontsize=22, loc='upper right')
    savefig(fig, 'NLBM_He.pdf')

if __name__ == "__main__":
    # Initial guess for the parameters
    params_H = [2.082, 2.780, 6.318, 0.193]
    plot_H(params_H)
    params_He = [0.133, 2.687, 5.930, 0.208]
    plot_He(params_He)
