import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
from scipy import integrate

from utils import set_axes, plot_data, savefig
from fit_phe import model_primary

import constants

params_H = [2.109, 2.794, 5.964, 0.190]
params_H_nobreak = [2.109, 2.794, 5.964, 0.]
params_He = [0.130, 2.655, 5.964 + np.log(2), 0.190]
params_He_nobreak = [0.130, 2.655, 5.964 + np.log(2), 0.]

def model_antimatter(energy, params, xsecs_interpolator, doBreak = True):
    pp_xsec = xsecs_interpolator["pp"]
    Hep_xsec = xsecs_interpolator["pHe"]

    XG, Xc, zeta, fCO = params

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

def plot_antimatter_comparison(params_NLBM, params_HM):
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

    from get_antimatter_xsecs import dsigmadE_ap, dsigmadE_pos
    ap_interpolators = dsigmadE_ap('linear')

    fudge = 1.4
    y_G, y_c = model_antimatter(E, params_NLBM, ap_interpolators, True)
    ax.plot(E, np.power(E, 2.5) * fudge * (y_G + y_c), color='tab:orange', lw=3, zorder=10, label='NLB model')
    ax.plot(E, np.power(E, 2.5) * fudge * y_G, color='tab:orange', ls='--', lw=3, zorder=10)

    y_HM = y_G / params_NLBM[0] * grammage_HM(E)

    ax.plot(E, np.power(E, 2.5) * fudge * y_HM, color='tab:red', lw=3, zorder=10, label='Halo model')

    # Show legend and save figure
    ax.legend(fontsize=22, loc='best')
    savefig(fig, 'NLBM_antimatter.pdf')

def plot_antimatter(params):
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    set_axes(ax, xlabel='E [GeV]', ylabel='E$^{2.8}$ I [GeV$^{1.8}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]', xscale='log', xlim=[2e1, 1e3], ylim=[0.1, 10.])
    
    # Plot data for positrons and antiprotons from AMS-02
    plot_data(ax, 'AMS-02_e+_Ek.txt', 2.8, 1., 'o', 'tab:blue', r'$e^+$', 3)
    plot_data(ax, 'AMS-02_pbar_Ek.txt', 2.8, 1., 'o', 'tab:red', r'$\bar p$', 3)
    
    # Generate and plot the model curve
    E = np.logspace(1, 3, 100) # GeV
    fudge = 3.0

    from get_antimatter_xsecs import dsigmadE_ap, dsigmadE_pos
    ap_interpolators = dsigmadE_ap('linear')

    y_G, y_c = model_antimatter(E, params, ap_interpolators, True)
    ax.plot(E, np.power(E, 2.5) * fudge * (y_G + y_c), color='tab:orange', lw=3, zorder=10)

    y_G, y_c = model_antimatter(E, params, ap_interpolators, False)
    ax.plot(E, np.power(E, 2.5) * fudge * (y_G + y_c), color='tab:orange', ls='--', lw=3, zorder=10)

    pos_interpolators = dsigmadE_pos('linear')

    from scipy.signal import savgol_filter

    y_G, y_c = model_antimatter(E, params, pos_interpolators, True)
    y = savgol_filter(np.power(E, 2.8) * (y_G + y_c), 15, 3)
    ax.plot(E, fudge * y, color='tab:purple', lw=3, zorder=10)

    y_G, y_c = model_antimatter(E, params, pos_interpolators, False)
    y = savgol_filter(np.power(E, 2.8) * (y_G + y_c), 15, 3)
    ax.plot(E, fudge * y, color='tab:purple', ls='--', lw=3, zorder=10)

    ax.fill_between([20., 60.3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)
    ax.fill_between([450., 1e3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)

    # Show legend and save figure
    ax.legend(fontsize=22, loc='upper left')
    savefig(fig, 'NLBM_antimatter.pdf')

# def plot_antimatter_ratio(params):
#     fig, ax = plt.subplots(figsize=(10.5, 8.5))
#     set_axes(ax, xlabel='E [GeV]', ylabel=r'$\bar p$/$e^+$', xlim=[2e1, 1e3], ylim=[0.25, 0.70])
    
#     plot_data(ax, 'AMS-02_pbar_e+_R.txt', 0., 1., 'o', 'tab:blue', r'AMS-02', 3)

#     # Generate and plot the model curve
#     E = np.logspace(1, 3, 200) # GeV

#     from get_antimatter_xsecs import dsigmadE_ap, dsigmadE_pos
    
#     ap_interpolators = dsigmadE_ap('linear')
#     pos_interpolators = dsigmadE_pos('linear')

#     y_G, y_c = model_antimatter(E, params, ap_interpolators, True)
#     y_ap = y_G + y_c

#     y_G, y_c = model_antimatter(E, params, pos_interpolators, True)
#     y_pos = y_G + y_c

#     y = y_ap / y_pos

#     from scipy.signal import savgol_filter

#     ax.plot(E, savgol_filter(y, 25, 3), color='tab:red')

#     y_G, y_c = model_antimatter(E, params, ap_interpolators, False)
#     y_ap = y_G + y_c

#     y_G, y_c = model_antimatter(E, params, pos_interpolators, False)
#     y_pos = y_G + y_c

#     y = y_ap / y_pos

#     ax.plot(E, savgol_filter(y, 21, 3), color='tab:red', ls='--')

#     ax.fill_between([20., 60.3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)
#     ax.fill_between([450., 1e3], 0., 10., color='tab:gray', alpha=0.15, zorder=1)

#     ax.hlines(0.479, 1e1, 1e3, color='tab:orange', ls='--')
#     ax.fill_between([1e1, 1e3], 0.479 - 0.014, 0.479 + 0.014, color='tab:orange', alpha=0.2, zorder=1)

#     ax.legend()
#     savefig(fig, 'NLBM_antimatter_ratio.pdf')

if __name__ == "__main__":
    # Initial guess for the parameters
    params_NLBM = [1.059, 2.671, 0.182, 0.853]
    params_HM = [4.751, 0.473, 251.901, 0.854]

    plot_antimatter_comparison(params_NLBM, params_HM)

    #plot_antimatter(params)
    #plot_antimatter_ratio(params)