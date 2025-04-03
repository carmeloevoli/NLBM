import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig
from fit_BC import model_BC, model_CO
from fit_BC_halo import model_BC_HM

def plot_NLBM_BC(params):
    """Plot B/C data and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel='B/C', xscale='log', xlim=[1e1, 1e4], ylim=[0.01, 0.22])

    # Plot data for B/C ratio from various sources
    plot_data(ax, 'AMS-02_BC_Ekn.txt', 0., 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'CALET_BC_Ekn.txt', 0., 1., 'o', 'tab:green', 'CALET', 3)
    plot_data(ax, 'DAMPE_BC_Ekn.txt', 0., 1., 'o', 'tab:brown', 'DAMPE', 2)

    # Generate and plot the model curve
    E = np.logspace(1, 4, 1000)
    y = model_BC(E, params)
    ax.plot(E, y, color='tab:red', lw=3.3, zorder=10)
    ax.hlines(y[-1], 1, 1e5, color='tab:red', linestyle=':', zorder=10)
    
    # Add text annotations
    ax.text(.9e3, 0.077, r'$\chi_{\rm G} = 1.1$ g/cm$^2$', color='tab:red', fontsize=25)
    ax.text(20, 0.20, r'$\chi_{\rm c,0} = 2.76$ g/cm$^2$', color='tab:red', fontsize=25)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper right')
    savefig(fig, 'NLBM_BC.pdf')

def plot_HM_BC(params):
    """Plot B/C data and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel='B/C', xscale='log', xlim=[1e1, 1e4], ylim=[0.01, 0.22])

    # Plot data for B/C ratio from various sources
    plot_data(ax, 'AMS-02_BC_Ekn.txt', 0., 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'CALET_BC_Ekn.txt', 0., 1., 'o', 'tab:green', 'CALET', 3)
    plot_data(ax, 'DAMPE_BC_Ekn.txt', 0., 1., 'o', 'tab:brown', 'DAMPE', 2)

    # Generate and plot the model curve
    E = np.logspace(1, 4, 1000)
    y = model_BC_HM(E, params)
    ax.plot(E, y, color='tab:red', lw=3.3, zorder=9)
    
    # Add text annotations
    ax.text(3e2, 0.087, r'$E_{\rm b} = 250$ GeV', color='tab:red', fontsize=25)
    ax.text(20, 0.20, r'$\chi_{G,0} = 4.7$ g/cm$^2$', color='tab:red', fontsize=25)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper right')
    savefig(fig, 'HM_BC.pdf')

def plot_CO(params):
    """Plot C/O data and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel='C/O', xscale='log', xlim=[10, 2.5e3], ylim=[0.7, 1.2])

    # Plot data for C/O ratio from various sources
    plot_data(ax, 'AMS-02_CO_R.txt', 0., 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'CALET_CO_Ekn.txt', 0., 1., 'o', 'tab:green', 'CALET', 2)

    # Generate and plot the model curve
    E = np.logspace(1, 4, 1000)
    y = model_CO(E, params)
    ax.plot(E, y, color='tab:red', lw=3, zorder=10)

    ax.text(20, 0.82, r'f$_{\text{C}/\text{O}} = 0.89$', color='tab:red', fontsize=25)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper left')
    savefig(fig, 'NLBM_CO.pdf')

def plot_grammage(params):
    def grammage(energy, A, params):
        XG, Xc0, zeta, fCO = params
        Xc = Xc0 * np.power(energy / 10., -zeta * np.log(energy))
        return XG + Xc
    
    def grammage_HM(energy, A, params):
        XG0, delta, Eb, fCO = params
        ddelta, s = delta, 0.1 
        XG = XG0 * np.power(energy / 10., -delta) 
        XG *= np.power(1. + np.power(energy / Eb, ddelta / s), s)
        return XG

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel=r'Grammage [g cm$^{-2}$]', xscale='log', xlim=[1e1, 1e4], ylim=[0.8, 5.0])

    E = np.logspace(1, 4, 1000)

    y = grammage(E, 10., params)
    ax.plot(E, y, color='tab:orange', ls='-', lw=4, zorder=8, label='NLBM')

    y = grammage_HM(E, 10., [4.751, 0.473, 251.901, 0.854])
    ax.plot(E, y, color='tab:blue', ls='--', lw=4, zorder=10, label='HM')

    # Show legend and save figure
    ax.legend(fontsize=25)
    savefig(fig, 'NLBM_grammage.pdf')

if __name__ == "__main__":
    # Initial guess for the parameters
    #params = [1.059, 2.671, 0.182, 0.853]
    params = [1.080, 2.758, 0.184, 0.894]
    plot_NLBM_BC(params)
    plot_CO(params)
    #params = [4.751, 0.473, 251.901, 0.854]
    #plot_HM_BC(params)
    #plot_grammage(params)
