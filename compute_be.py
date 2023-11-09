import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math
#from dataclasses import dataclass
#
#from scipy.integrate import quad
#from scipy.interpolate import RegularGridInterpolator

# FIXED PARAMS

year = 3.1e7 # s
Myr = 1e6 * year # s
pc = 3.1e18 # cm
kpc = 1e3 * pc # cm
cLight = 3e10 # cm / s
snrate = 1. / 50 / year # 1/s
hdisk = 100. * pc # cm
R_G = 10. * kpc # cm
V_G = math.pi * np.power(R_G, 2.) * 2. * hdisk # cm^3
E_0 = 10. # GeV
sigmaCB = 60e-27 # cm^2
n_H = 1. # 1/cm^3
n_c = 50. # 1/cm^3
E_CR = 1e51 # GeV
mbarn = 1e-27 # cm^2

def plot_data(ax, filename, zorder=1):
    fmt = 'o'
    label = 'AMS-02'
    color = 'tab:red'
    E, y = np.loadtxt(filename,usecols=(0,1),unpack=True)
    y_err_lo = 0.1 * y
    y_err_up = 0.1 * y
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=zorder)


#@dataclass
#class InjectionParams:
#    A: float
#    lnEb: float
#    p: float
#    dp: float
#    s: float
#    tauG: float
#
#@dataclass
#class GrammageParams:
#    tauG: float
#    tau0: float
#    zeta: float
#
#def primaryModel(E, par):
#    value = snrate * par.tauG / V_G # cm^-3
#    value *= E_CR / np.power(E_0, 2.) * (par.p - 2.) # GeV^-1 cm^-3
#    value *= cLight / 4. / math.pi # GeV^-1 cm^-2 s^-1 sr^-1
#    Eb = np.exp(par.lnEb)
#    y = par.A * value * (E / E_0)**(-par.p)
#    y *= np.power(1. + np.power(E / Eb, par.dp / par.s), par.s)
#    return y
#
#def I_ap():
#    Ip_par = InjectionParams(14.864, np.log(639.935), 2.811, 0.251, 0.123, 0.937 * Myr)
#    Ihe_par = InjectionParams(0.948, np.log(474.829), 2.737, 0.308, 0.192, 0.937 * Myr)
#
#    chi_par = GrammageParams(0.937 * Myr, 0.079 * Myr, 0.080)
#    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = get_dsigmadE_ap()
#
#    def integrand_qg(lnEprime, E):
#        Eprime = np.exp(lnEprime)
#        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        return Eprime * chi_par.tauG * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)
#
#    def integrand_qc(lnEprime, E):
#        Eprime = np.exp(lnEprime)
#        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        tau_c = chi_par.tau0 * np.power(Eprime, -chi_par.zeta * np.log(Eprime))
#        return Eprime * tau_c * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)
#
#    size = 100
#    E = np.logspace(0, 4, size)
#    value_G = np.zeros(size)
#    for i in range(size):
#        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
#    value_G *= cLight * n_H
#
#    value_c = np.zeros(size)
#    for i in range(size):
#        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
#    value_c *= cLight * n_c
#
#    return E, value_c, value_G
#
#def I_pos():
#    Ip_par = InjectionParams(14.864, np.log(639.935), 2.811, 0.251, 0.123, 0.937 * Myr)
#    Ihe_par = InjectionParams(0.948, np.log(474.829), 2.737, 0.308, 0.192, 0.937 * Myr)
#
#    chi_par = GrammageParams(0.937 * Myr, 0.079 * Myr, 0.080)
#    pp_xsec, pHe_xsec, Hep_xsec, HeHe_xsec = gest_dsigmadE_pos()
#
#    def integrand_qg(lnEprime, E):
#        Eprime = np.exp(lnEprime)
#        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        return Eprime * chi_par.tauG * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)
#
#    def integrand_qc(lnEprime, E):
#        Eprime = np.exp(lnEprime)
#        pp_s = pp_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        hep_s = Hep_xsec([np.log10(E), np.log10(Eprime)]) * mbarn
#        tau_c = chi_par.tau0 * np.power(Eprime, -chi_par.zeta * np.log(Eprime))
#        return Eprime * tau_c * (primaryModel(Eprime, Ip_par) * pp_s + primaryModel(Eprime, Ihe_par) * hep_s)
#
#    size = 100
#    E = np.logspace(0, 4, size)
#    value_G = np.zeros(size)
#    for i in range(size):
#        value_G[i] = quad(integrand_qg, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
#    value_G *= cLight * n_H
#
#    value_c = np.zeros(size)
#    for i in range(size):
#        value_c[i] = quad(integrand_qc, np.log(E[i]), np.log(1e4 * E[i]), args=(E[i]))[0]
#    value_c *= cLight * n_c
#
#    return E, value_c, value_G
    
def plot_be():
    def set_axes(ax):
        ax.set_xlabel('E [GeV/n]')
        ax.set_xlim([0.01, 12])
#        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_ylim([0., 0.5])
#        ax.set_yscale('log')

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    #ax.fill_between([10,20], 0, 7, color='tab:gray', alpha=0.2)

    plot_data(ax, 'data/be10be9.txt')
    
    E = np.linspace(0, 12, 1000)
    
    f = 0.315
    tau_G = 1. * Myr
    tau_d = 2. * Myr * (E / 1.)
    ax.plot(E, f * tau_d / (tau_G + tau_d))

    f = 0.35
    tau_G = 3. * Myr
    ax.plot(E, f * tau_d / (tau_G + tau_d), linestyle=':')

    f = 0.43
    tau_G = 9. * Myr
    ax.plot(E, f * tau_d / (tau_G + tau_d), linestyle=':')


    ax.legend(fontsize=30, loc='best')
    plt.savefig('NLBM_be.pdf')
  
if __name__== "__main__":
    plot_be()
