import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig

def plot_Be_ratio(params):
    """Plot Be ratio and model."""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV/n]', ylabel='$^{10}$Be/$^9$Be', xlim=[0.3, 2e1], ylim=[0.06, 0.4])

    filename = '../data/Be10_Be9_AMS02_TeVPA2024.txt'
    E, r, r_min, r_max = np.loadtxt(filename, usecols=(0,1,2,3), unpack=True)

    color = 'r'
    ax.errorbar(E, r, yerr=[r - r_min, r_max - r], fmt='o', markeredgecolor=color, color=color, 
                label='AMS-02 (preliminary)', capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=2)

    # # Plot data for B/C ratio from various sources
    # plot_data(ax, 'AMS-02_BC_Ekn.txt', 0., 1., 'o', 'tab:orange', 'AMS-02', 1)
    # plot_data(ax, 'DAMPE_BC_Ekn.txt', 0., 1., 'o', 'r', 'CALET', 3)
    # plot_data(ax, 'CALET_BC_Ekn.txt', 0., 1., 'o', 'g', 'DAMPE', 2)

    # # Generate and plot the model curve
    # E = np.logspace(1, 4, 1000)
    # y = model_BC(E, params)
    # ax.plot(E, y, color='b', lw=3, zorder=10)
    # ax.hlines(y[-1], 1, 1e5, color='b', linestyle='--', zorder=10)

    # # Add text annotations
    # ax.text(.9e3, 0.077, r'$\chi_{\rm G} = 0.6$ gr/cm$^2$', color='b', fontsize=25)
    # ax.text(20, 0.20, r'$\chi_{\rm 0} = 3$ gr/cm$^2$', color='b', fontsize=25)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper right')
    savefig(fig, 'NLBM_Be10_TeVPA2024.pdf')

if __name__ == "__main__":
    # Initial guess for the parameters
    params = [0.648, 2.871, 0.091, 0.897]
    plot_Be_ratio(params)
