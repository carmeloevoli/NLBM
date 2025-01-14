import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
from scipy import integrate

from utils import set_axes, plot_data, savefig
from fit_phe import model_primary

import constants

params_BC = [1.080, 2.758, 0.184, 0.894]
params_HM = [4.751, 0.473, 251.901, 0.854]

params_H = [2.082, 2.780, 6.318, 0.183]
params_H_nobreak = [2.082, 2.780, 6.318, 0.]
params_He = [0.133, 2.687, 5.930, 0.208]
params_He_nobreak = [0.133, 2.687, 5.930, 0.]

def model_antiprotons(energy, doBreak = True):
    from get_antimatter_xsecs import dsigmadE_ap, dsigmadE_pos
    ap_interpolators = dsigmadE_ap('cubic')
    pp_xsec = ap_interpolators["pp"]
    Hep_xsec = ap_interpolators["pHe"]

    XG, Xc, zeta, fCO = params_BC

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_xs = pp_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        hep_xs = Hep_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        if doBreak:
            I_H = model_primary(Eprime, params_H)
            I_He = model_primary(Eprime, params_He)
        else:
            I_H = model_primary(Eprime, params_H_nobreak)
            I_He = model_primary(Eprime, params_He_nobreak)            
        return Eprime * XG * (I_H * pp_xs + I_He * hep_xs)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_xs = pp_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        hep_xs = Hep_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        X_c = Xc * np.power(Eprime, -zeta * np.log(Eprime))
        if doBreak:
            I_H = model_primary(Eprime, params_H)
            I_He = model_primary(Eprime, params_He)
        else:
            I_H = model_primary(Eprime, params_H_nobreak)
            I_He = model_primary(Eprime, params_He_nobreak)            
        return Eprime * X_c * (I_H * pp_xs + I_He * hep_xs)

    size = len(energy)
    
    value_G = np.zeros(size)
    for i in range(size):
        lgEmin = np.log(energy[i])
        lgEmax = np.log(1e4 * energy[i])
        value_G[i] = integrate.quad(integrand_qg, lgEmin, lgEmax, args=(energy[i]))[0]
    value_G /= constants.PROTON_MASS
    
    value_c = np.zeros(size)
    for i in range(size):
        lgEmin = np.log(energy[i])
        lgEmax = np.log(1e4 * energy[i])
        value_c[i] = integrate.quad(integrand_qc, lgEmin, lgEmax, args=(energy[i]))[0]
    value_c /= constants.PROTON_MASS
    
    return value_G, value_c

def model_positrons(energy, doBreak = True, doLosses = True):
    from get_antimatter_xsecs import dsigmadE_ap, dsigmadE_pos
    pos_interpolators = dsigmadE_pos('cubic')
    pp_xsec = pos_interpolators["pp"]
    Hep_xsec = pos_interpolators["pHe"]

    XG, Xc, zeta, fCO = params_BC

    def integrand_qg(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_xs = pp_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        hep_xs = Hep_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        if doBreak:
            I_H = model_primary(Eprime, params_H)
            I_He = model_primary(Eprime, params_He)
        else:
            I_H = model_primary(Eprime, params_H_nobreak)
            I_He = model_primary(Eprime, params_He_nobreak)            
        return Eprime * XG * (I_H * pp_xs + I_He * hep_xs)

    def integrand_qc(lnEprime, E):
        Eprime = np.exp(lnEprime)
        pp_xs = pp_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        hep_xs = Hep_xsec([np.log10(Eprime), np.log10(E)])[0] * constants.MBARN
        X_c = Xc * np.power(Eprime, -zeta * np.log(Eprime))
        if doBreak:
            I_H = model_primary(Eprime, params_H)
            I_He = model_primary(Eprime, params_He)
        else:
            I_H = model_primary(Eprime, params_H_nobreak)
            I_He = model_primary(Eprime, params_He_nobreak)            
        return Eprime * X_c * (I_H * pp_xs + I_He * hep_xs)

    size = len(energy)
    
    value_G = np.zeros(size)
    for i in range(size):
        lgEmin = np.log(energy[i])
        lgEmax = np.log(1e4 * energy[i])
        value_G[i] = integrate.quad(integrand_qg, lgEmin, lgEmax, args=(energy[i]))[0]
    value_G /= constants.PROTON_MASS
    
    value_c = np.zeros(size)
    for i in range(size):
        lgEmin = np.log(energy[i])
        lgEmax = np.log(1e4 * energy[i])
        value_c[i] = integrate.quad(integrand_qc, lgEmin, lgEmax, args=(energy[i]))[0]
    value_c /= constants.PROTON_MASS
    
    tau_G = 1. # Myr
    tau_loss = 0.5 * (1e3 / energy) # Myr

    if doLosses:
        y = np.power(1. + tau_G / tau_loss, -1.) 
    else:
        y = 1.

    return value_G * y, value_c * y

def plot_antimatter():
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV]', ylabel='E$^{2.8}$ I [GeV$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]', xscale='log', xlim=[2e1, 1e3], ylim=[0.1, 10.])
    
    # Plot data for positrons and antiprotons from AMS-02
    plot_data(ax, 'AMS-02_e+_Ek.txt', 2.8, 1., 'o', 'tab:blue', r'$e^+$', 3)
    plot_data(ax, 'AMS-02_pbar_Ek.txt', 2.8, 1., 'o', 'tab:orange', r'$\bar p$', 3)
    
    # Generate and plot the model curve
    E = np.logspace(1, 3, 100) # GeV
    fudge = 2.1

    # Solid: with break and losses
    y_G, y_c = model_antiprotons(E, doBreak=True)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:red', lw=3, zorder=10)

    y_G, y_c = model_positrons(E, doBreak=True, doLosses=True)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:purple', lw=3, zorder=10)

    # Dashed: with break and no losses
    y_G, y_c = model_antiprotons(E, doBreak=True)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:red', lw=3, zorder=10, ls='--')

    y_G, y_c = model_positrons(E, doBreak=True, doLosses=False)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:purple', lw=3, zorder=10, ls='--')

    # Dotted: no break and no losses
    y_G, y_c = model_antiprotons(E, doBreak=False)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:red', lw=3, zorder=10, ls=':')

    y_G, y_c = model_positrons(E, doBreak=False, doLosses=False)
    ax.plot(E, np.power(E, 2.8) * fudge * (y_G + y_c), color='tab:purple', lw=3, zorder=10, ls=':')

    ax.fill_between([20., 60.3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)
    ax.fill_between([450., 1e3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper left')
    savefig(fig, 'NLBM_antimatter.pdf')

def plot_antimatter_ratio():
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV]', ylabel=r'$\bar p$/$e^+$', xlim=[2e1, 5e2], ylim=[0.25, 0.75])
    
    plot_data(ax, 'AMS-02_pbar_e+_R.txt', 0., 1., 'o', 'tab:blue', r'AMS-02', 3)

    # Generate and plot the model curve
    E = np.logspace(1, 3, 200) # GeV

    # Solid: with break and losses
    y_G, y_c = model_antiprotons(E, doBreak=True)
    y_ap = y_G + y_c

    y_G, y_c = model_positrons(E, doBreak=True, doLosses=True)
    y_pos = y_G + y_c

    ax.plot(E, y_ap / y_pos, color='tab:red')

    # Dashed: with break and no losses
    y_G, y_c = model_antiprotons(E, doBreak=True)
    y_ap = y_G + y_c

    y_G, y_c = model_positrons(E, doBreak=True, doLosses=False)
    y_pos = y_G + y_c

    ax.plot(E, y_ap / y_pos, color='tab:red', ls='--')

    # Dotted: no break and no losses
    y_G, y_c = model_antiprotons(E, doBreak=False)
    y_ap = y_G + y_c

    y_G, y_c = model_positrons(E, doBreak=False, doLosses=False)
    y_pos = y_G + y_c

    ax.plot(E, y_ap / y_pos, color='tab:red', ls=':')

    ax.fill_between([20., 60.3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)
    ax.fill_between([450., 1e3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)

    ax.hlines(0.479, 1e1, 1e3, color='tab:orange', ls='--')
    ax.fill_between([1e1, 1e3], 0.479 - 0.014, 0.479 + 0.014, color='tab:orange', alpha=0.2, zorder=1)

    ax.legend()
    savefig(fig, 'NLBM_antimatter_ratio.pdf')

def plot_antimatter_comparison():
    def grammage_HM(energy):
        XG0, delta, Eb, fCO = params_HM
        ddelta, s = delta, 0.1
        XG = XG0 * np.power(energy / 10., -delta) 
        XG *= np.power(1. + np.power(energy / Eb, ddelta / s), s)
        return XG
    
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV]', ylabel='E$^{2.5}$ I [GeV$^{1.5}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]', xscale='log', xlim=[2e1, 1e3], ylim=[0.2, 1.4])

    plot_data(ax, 'AMS-02_pbar_Ek.txt', 2.5, 1., 'o', 'tab:blue', r'AMS-02', 3)

    # Generate and plot the model curve
    E = np.logspace(1, 3, 100) # GeV

    fudge = 1.4
    y_G, y_c = model_antiprotons(E, True)
    ax.plot(E, np.power(E, 2.5) * fudge * (y_G + y_c), color='tab:orange', lw=3, zorder=10, label='NLB model')
    ax.plot(E, np.power(E, 2.5) * fudge * y_G, color='tab:orange', ls='--', lw=3, zorder=10)

    y_HM = y_G / params_BC[0] * grammage_HM(E)

    ax.plot(E, np.power(E, 2.5) * fudge * y_HM, color='tab:red', lw=3, zorder=10, label='Halo model')

    # Show legend and save figure
    ax.legend(fontsize=22, loc='best')
    savefig(fig, 'NLBM_antimatter_HM.pdf')

if __name__ == "__main__":
    #plot_antimatter()
    #plot_antimatter_ratio()
    plot_antimatter_comparison()
