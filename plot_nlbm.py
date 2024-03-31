import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from fit_nlbm import GrammageParams, InjectionParams
from fit_nlbm import fit_BC, fit_H, fit_He, model_CO, model_Be_ratio, model_primary, model_antiprotons, model_positrons
from fit_nlbm import test_xsecs, compute_tauG, compute_tauLoss, Myr

def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    x, y, err_tot_lo, err_tot_up = np.loadtxt(filename,usecols=(0,1,2,3),unpack=True)
    x = x / norm
    y = norm * np.power(x, slope) * y
    y_err_lo = norm * np.power(x, slope) * err_tot_lo
    y_err_up = norm * np.power(x, slope) * err_tot_up
    ax.errorbar(x, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)
                
def plot_BC(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'B/C')
        ax.set_ylim([0.01, 0.23])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'data/AMS-02_BC.txt', 0., 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/DAMPE_BC.txt', 0., 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/CALET_BC.txt', 0., 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, y_G, values = fit_BC(initialValues)
    ax.plot(E, y, color='tab:blue', zorder=10)

    ax.hlines(y_G, 1, 1e5, color='tab:blue', linestyle='--', zorder=10)
    ax.text(1.2e3, 0.075, r'$\chi_{\rm G} = 1$ gr/cm$^2$', color='tab:blue', fontsize=22)
    ax.text(20, 0.20, r'$\chi_{\rm 0} = 4.5$ gr/cm$^2$', color='tab:blue', fontsize=22)

    ax.legend(fontsize=20, loc='upper right')
    plt.savefig('NLBM_BC.pdf')

    return values
        
def plot_CO(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'C/O')
        ax.set_ylim([0.65, 1.2])
   
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'data/AMS-02_CO.txt', 0., 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_CO.txt', 0., 1., 'o', 'limegreen', 'CALET', 2)

    E = np.logspace(1, 4, 100)
    y = model_CO(E, initialValues)
    ax.plot(E, y, color='tab:blue', zorder=10)

    ax.hlines(initialValues.fCO, 1, 1e5, color='tab:blue', linestyle='--', zorder=10)
    #ax.text(20., 0.91, r'$q_C / q_O = 0.9$', color='tab:blue', fontsize=22)
    ax.legend(fontsize=20, loc='upper right')
    plt.savefig('NLBM_CO.pdf')

def plot_H(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.8}$ I [GeV$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([1e4, 4e4])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
        
    plot_data(ax, 'data/AMS-02_H.txt', 2.8, 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_H.txt', 2.8, 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/DAMPE_H.txt', 2.8, 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, values = fit_H(initialValues)
    ax.plot(E, np.power(E, 2.8) * y, color='tab:blue', zorder=10)
    
    y = model_primary(E, values, False)
    ax.plot(E, np.power(E, 2.8) * y, color='tab:blue', linestyle='--', zorder=10)
    
    ax.legend(fontsize=20, loc='upper left')
    plt.savefig('NLBM_H.pdf')
    
    return values
    
def plot_He(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e4])
        ax.set_ylabel(r'E$^{2.8}$ I [GeV/n$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([0.5e3, 5e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    values = []
    
    plot_data(ax, 'data/AMS-02_He.txt', 2.8, 1., 'o', 'crimson', 'AMS-02', 1)
    plot_data(ax, 'data/CALET_He.txt', 2.8, 1., 'o', 'limegreen', 'CALET', 2)
    plot_data(ax, 'data/DAMPE_He.txt', 2.8, 1., 'o', 'darkorange', 'DAMPE', 3)

    E, y, values = fit_He(initialValues)
    ax.plot(E, np.power(E, 2.8) * y, color='tab:blue', zorder=10)

    y = model_primary(E, values, False)
    ax.plot(E, np.power(E, 2.8) * y, color='tab:blue', linestyle='--', zorder=10)

    ax.legend(fontsize=20, loc='upper left')
    plt.savefig('NLBM_He.pdf')

    return values
    
def plot_Be(initialValues):
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xlim([0, 12])
        ax.set_ylabel(r'Be$^{10}$ / Be$^9$')
        ax.set_ylim([0., 0.75])
#        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    E = np.linspace(0, 20, 1000)

    r = model_Be_ratio(E, 1. * Myr, 0.9)
    ax.plot(E, r, color='tab:blue', label='1 Myr')

    r = model_Be_ratio(E, 5 * Myr, 0.9)
    ax.plot(E, r, color='tab:orange', label='5 Myr')

    r = model_Be_ratio(E, 20 * Myr, 0.9)
    ax.plot(E, r, color='tab:green', label='20 Myr')

#    sigma_10 = sigma_C10 + f_OC * sigma_O10
#    sigma_9 = sigma_C9 + f_OC * sigma_O9
#
#    ax.hlines(sigma_10 / sigma_9, 0, 20, ls='--', color='tab:blue')
#    ax.text(8., 0.67, r'$\sigma_{10}/\sigma_9$', color='tab:blue')
    
    ax.text(1, 0.05, 'AMS-02 (2023) preliminary', fontsize=20)
    
    ax.legend(fontsize=18, loc='lower right')
    
    plot_data(ax, 'data/AMS-02_Be_ratio_preliminary.txt', 0., 1., 'o', 'crimson', 'AMS-02', 1)

    plt.savefig('NLBM_Be_ratio.pdf')
    
def plot_antimatter(grammage_params, H_params, He_params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{2.8}$ I [GeV$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([0., 12.])
        #ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'data/AMS-02_pbar.txt', 2.8, 1., 'o', 'tab:red', r'$\bar p$', 1)
    plot_data(ax, 'data/AMS-02_pos.txt', 2.8, 1., 'o', 'tab:blue', r'$e^+$', 1)

    E = np.logspace(1, 3, 200)
    
    f_ap = 2.5
    
    I_G, I_c = model_antiprotons(E, grammage_params, H_params, He_params)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'darkorange', linestyle='-')

    I_G, I_c = model_antiprotons(E, grammage_params, H_params, He_params, False)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'darkorange', linestyle='--')

    I_G, I_c = model_positrons(E, grammage_params, H_params, He_params)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'tab:purple', linestyle='-')

    I_G, I_c = model_positrons(E, grammage_params, H_params, He_params, False)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'tab:purple', linestyle='--')

    ax.legend(fontsize=26, loc='best')
    print('Saving NLBM_antimatter.pdf')
    plt.savefig('NLBM_antimatter.pdf')

def plot_antimatter_cocoon(grammage_params, H_params, He_params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{2.8}$ I [GeV$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_ylim([0., 12.])
        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'data/AMS-02_pbar.txt', 2.8, 1., 'o', 'tab:red', r'$\bar p$', 1)
    plot_data(ax, 'data/AMS-02_pos.txt', 2.8, 1., 'o', 'tab:blue', r'$e^+$', 1)

    E = np.logspace(1, 3, 200)
    
    f_ap = 2.5
    
    I_G, I_c = model_antiprotons(E, grammage_params, H_params, He_params)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'darkorange', linestyle='-')
    ax.plot(E, np.power(E, 2.8) * f_ap * I_G, 'darkorange', linestyle='--')
    ax.plot(E, np.power(E, 2.8) * f_ap * I_c, 'darkorange', linestyle=':')

    I_G, I_c = model_positrons(E, grammage_params, H_params, He_params)
    ax.plot(E, np.power(E, 2.8) * f_ap * (I_G + I_c), 'tab:purple', linestyle='-')
    ax.plot(E, np.power(E, 2.8) * f_ap * I_G, 'tab:purple', linestyle='--')
    ax.plot(E, np.power(E, 2.8) * f_ap * I_c, 'tab:purple', linestyle=':')

    ax.legend(fontsize=26, loc='best')
    print('Saving NLBM_antimatter_coccon.pdf')
    plt.savefig('NLBM_antimatter_coccon.pdf')

def plot_antimatter_ratio(grammage_params, H_params, He_params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        #ax.set_xscale('log')
        ax.set_xlim([2e1, 400])
        ax.set_ylabel(r'$\bar p$ / $e^+$')
        ax.set_ylim([0., 0.8])
        #ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'data/AMS-02_pbar_pos.txt', 0., 1., 'o', 'tab:red', r'$\bar p$', 1)

    E = np.linspace(10, 500, 100)

    I_ap_G, I_ap_c = model_antiprotons(E, grammage_params, H_params, He_params)
    I_pos_G, I_pos_c = model_positrons(E, grammage_params, H_params, He_params)
    ax.plot(E, (I_ap_G + I_ap_c) / (I_pos_G + I_pos_c))
    
    I_ap_G, I_ap_c = model_antiprotons(E, grammage_params, H_params, He_params, False)
    I_pos_G, I_pos_c = model_positrons(E, grammage_params, H_params, He_params, False)
    ax.plot(E, (I_ap_G + I_ap_c) / (I_pos_G + I_pos_c))

    ax.legend(fontsize=26, loc='best')
    print('Saving NLBM_antimatter_ratio.pdf')
    plt.savefig('NLBM_antimatter_ratio.pdf')
    
def plot_test_xsecs():
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e1, 1e3])
        ax.set_ylabel(r'E$^{\alpha}$ Q($\alpha$)')
        ax.set_ylim([0.2, 1.2])
        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    E = np.logspace(1, 3, 100)
    
    y = test_xsecs(E, 2.4)
    ax.plot(E, y / max(y), label=r'$\alpha = 2.4$')

    y = test_xsecs(E, 2.6)
    ax.plot(E, y / max(y), label=r'$\alpha = 2.6$')

    y = test_xsecs(E, 2.8)
    ax.plot(E, y / max(y), label=r'$\alpha = 2.8$')

    ax.plot(E, np.power(E / 1e3, 0.14), ':')

    ax.legend(fontsize=24, loc='best')
    print('Saving NLBM_test_xsecs.pdf')
    plt.savefig('NLBM_test_xsecs.pdf')

if __name__== "__main__":
    grammage_params = GrammageParams(1., 4.5, 0.1, 0.9)
    grammage_params = plot_BC(GrammageParams(1., 4.5, 0.1, 0.9))
    plot_CO(grammage_params)
    tau_G = compute_tauG(grammage_params)
    print(f'tau_G : {tau_G/Myr:5.2f} Myr')
    H_params = InjectionParams(0.034, np.log(580.), 2.80, 0.23, 0.05, tau_G)
    H_params = plot_H(H_params)
    He_params = InjectionParams(0.002, np.log(380.), 2.72, 0.25, 0.05, tau_G)
    He_params = plot_He(He_params)
    plot_Be(grammage_params)
    tau_loss = compute_tauLoss(1e3)
    print(f'tau_loss(TeV) : {tau_loss/Myr:5.2f} Myr')
    plot_antimatter(grammage_params, H_params, He_params)
    plot_antimatter_cocoon(grammage_params, H_params, He_params)
    plot_antimatter_ratio(grammage_params, H_params, He_params)
    plot_test_xsecs()
