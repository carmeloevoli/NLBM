import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np
import math

#from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

def dsigmadE_ap(method='linear'):
    Eproj_table, Epbar_table, pp_table, pHe_table, Hep_table, HeHe_table = np.loadtxt('tables/supplementary__XS_table_Param_II_B.txt', usecols=(0,1,2,3,8,9), unpack=True)
    projectile_kinetic_energy = np.linspace(np.log10(1.), np.log10(1e7), 211)
    secondary_kinetic_energy = np.linspace(np.log10(0.1), np.log10(1e4), 151)
    points = (secondary_kinetic_energy, projectile_kinetic_energy)

    pp_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    pHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    Hep_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    HeHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    count = 0
    for j in range(len(projectile_kinetic_energy)):
        pp_xsec[:,j] = pp_table[count:count + 151] * 1e31 # m2 -> mbarn
        pHe_xsec[:,j] = pHe_table[count:count + 151] * 1e31 # m2 -> mbarn
        Hep_xsec[:,j] = Hep_table[count:count + 151] * 1e31 # m2 -> mbarn
        HeHe_xsec[:,j] = HeHe_table[count:count + 151] * 1e31 # m2 -> mbarn
        count = count + 151
    pp_i = RegularGridInterpolator(points, pp_xsec, method=method)
    pHe_i = RegularGridInterpolator(points, pHe_xsec, method=method)
    Hep_i = RegularGridInterpolator(points, Hep_xsec, method=method)
    HeHe_i = RegularGridInterpolator(points, HeHe_xsec, method=method)

    return pp_i, pHe_i, Hep_i, HeHe_i

def dsigmadE_pos(method='linear'):
    Eproj_table, Epbar_table, pp_table, pHe_table, Hep_table, HeHe_table = np.loadtxt('tables/supplementary_table_positrons_best_fit.dat', usecols=(0,1,2,3,8,9), unpack=True)

    projectile_kinetic_energy = np.linspace(np.log10(0.1), np.log10(1e6), 140)
    secondary_kinetic_energy = np.linspace(np.log10(0.01), np.log10(1e4), 90)
    points = (secondary_kinetic_energy, projectile_kinetic_energy)

    pp_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    pHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    Hep_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    HeHe_xsec = np.zeros((len(secondary_kinetic_energy), len(projectile_kinetic_energy)))
    count = 0
    for j in range(len(projectile_kinetic_energy)):
        pp_xsec[:,j] = pp_table[count:count + 90] # mbarn
        pHe_xsec[:,j] = pHe_table[count:count + 90] # mbarn
        Hep_xsec[:,j] = Hep_table[count:count + 90] # mbarn
        HeHe_xsec[:,j] = HeHe_table[count:count + 90] # mbarn
        count = count + 90
    pp_i = RegularGridInterpolator(points, pp_xsec, method=method)
    pHe_i = RegularGridInterpolator(points, pHe_xsec, method=method)
    Hep_i = RegularGridInterpolator(points, Hep_xsec, method=method)
    HeHe_i = RegularGridInterpolator(points, HeHe_xsec, method=method)

    return pp_i, pHe_i, Hep_i, HeHe_i

def test_apxsec():
    def set_axes(ax):
        ax.set_xlabel('x')
        ax.set_xscale('log')
        ax.set_xlim([1e-4, 1])
        ax.set_ylabel(r'x$^2$ d$\sigma$/dx [mbarn]')
        ax.set_yscale('log')
        ax.set_ylim([1e-3, 1e1])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    dsdE_pp, dsdE_phe, dsdE_hep, dsdE_hehe = dsigmadE_pos('cubic')

    Ep = 1e3
    x = np.logspace(-4, 0, 1000)
    s = []
    for x_i in x:
        s.append(dsdE_pp([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:orange')

    s = []
    for x_i in x:
        s.append(dsdE_phe([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:red')

    s = []
    for x_i in x:
        s.append(dsdE_hep([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:green')

    s = []
    for x_i in x:
        s.append(dsdE_hehe([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:purple')

    #
    
    dsdE_pp, dsdE_phe, dsdE_hep, dsdE_hehe = dsigmadE_ap('cubic')

    Ep = 1e3
    x = np.logspace(-4, 0, 1000)
    s = []
    for x_i in x:
        s.append(dsdE_pp([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:orange', linestyle=':')

    s = []
    for x_i in x:
        s.append(dsdE_phe([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:red', linestyle=':')

    s = []
    for x_i in x:
        s.append(dsdE_hep([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:green', linestyle=':')

    s = []
    for x_i in x:
        s.append(dsdE_hehe([np.log10(x_i * Ep), np.log10(Ep)])[0])
    ax.plot(x, x * x * np.array(s) * Ep, color='tab:purple', linestyle=':')
    
    plt.savefig('antimatter_xsecs.pdf')

if __name__== "__main__":
    test_apxsec()
