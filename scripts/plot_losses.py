import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig
from constants import ME_C2, SIGMA_THOMSON, C_LIGHT, K_BOLTZMANN, ERG_TO_GEV, MYR

def dEdt_Thomson(E, energy_density):
    gamma_e = E / ME_C2 
    b_0 = 4.0 / 3.0 * SIGMA_THOMSON * C_LIGHT * energy_density
    return b_0 * np.power(gamma_e, 2.0)


def dEdt_KN(E, T, energy_density):
    def Y(x :float) -> float:
        if x < 1.5e-3:
            return np.power(np.pi, 4) / 15.0
        elif x < 150.0:
            c = [-3.996e-2, -9.100e-1, -1.197e-1, 3.305e-3,
                1.044e-3, -7.013e-5, -9.618e-6]
            c_log = sum(c[i] * np.power(np.log(x), i) for i in range(7))
            return np.exp(c_log)
        else:
            return 3.0 / 4.0 * np.power(np.pi / x, 2) * (np.log(x) - 1.9805)
    gamma_e = E / ME_C2
    factor = 20. * C_LIGHT * SIGMA_THOMSON / np.power(np.pi, 4)
    S_i_ = []
    for i in gamma_e:
        if T < 0.:
            S_i_.append(np.power(np.pi, 4) / 15.0)
        else:
            S_i_.append(Y(4. * i * K_BOLTZMANN * T / ME_C2))
    S_i_ = np.array(S_i_)

    return factor * energy_density * S_i_ * np.power(gamma_e, 2.0)

def tau_loss(E, B):
    U_B = B**2 / (8 * np.pi) * ERG_TO_GEV  # Energy density in GeV/cm^3
    tau_B = E / dEdt_Thomson(E, U_B)
    tau_CMB = E / dEdt_KN(E, 2.7, 0.25e-9)
    tau_IR = E / dEdt_KN(E, 33.07, 25.4e-11)
    tau_opt = E / dEdt_KN(E, 313.32, 5.47e-11)
    tau_UV1 = E / dEdt_KN(E, 3249.3, 37e-11)
    tau_UV2 = E / dEdt_KN(E, 6150.4, 22.9e-11)
    tau_UV3 = E / dEdt_KN(E, 23209.0, 11.89e-11)
    tau_ISRF = 1. / (1. / tau_B + 1. / tau_CMB + 1. / tau_IR + 1. / tau_opt + 1. / tau_UV1 + 1. / tau_UV2 + 1. / tau_UV3)
    return tau_ISRF

def plot_losses():
    """Plot energy losses for different models."""
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    set_axes(ax, xlabel='E [GeV]', ylabel=r'$\tau$ [Myr]', xscale='log', yscale='log', xlim=[1e1, 1e4], ylim=[1e-2, 1e3])

    # Define parameters
    B_FIELD = 1e-6  # Magnetic field strength in gauss
    U_B = B_FIELD**2 / (8 * np.pi) * ERG_TO_GEV  # Energy density in GeV/cm^3
    U_CMB = 0.25e-9  # Energy density in GeV/cm^3
    U_ISRF = 0e-9  # Energy density in GeV/cm^3

    print(f'{U_B:.2e}, {U_CMB:.2e}, {U_ISRF:.2e}')

    # Generate energy values
    E = np.logspace(1, 6, 1000) # GeV

    # Calculate energy losses for different models
    tau_B = E / dEdt_Thomson(E, U_B)
    tau_ISRF = E / dEdt_Thomson(E, U_CMB) 

    # Plot the results for Thomson
    ax.plot(E, tau_B / MYR, color='tab:red', lw=2.3, ls='--', zorder=10, label=f'B = 1 $\mu$G')
    ax.plot(E, tau_ISRF / MYR, color='tab:orange', lw=2.3, ls='--', zorder=10, label=f'CMB Thomson')
    tau_all = 1. / (1. / tau_B + 1. / tau_ISRF)
    ax.plot(E, tau_all / MYR, color='tab:blue', lw=3.3, zorder=10, label=f'B + CMB Thomson')

    # Calculate energy losses for different models using Klein-Nishina
    tau_CMB = E / dEdt_KN(E, 2.7, 0.25e-9)
    tau_IR = E / dEdt_KN(E, 33.07, 25.4e-11)
    tau_opt = E / dEdt_KN(E, 313.32, 5.47e-11)
    tau_UV1 = E / dEdt_KN(E, 3249.3, 37e-11)
    tau_UV2 = E / dEdt_KN(E, 6150.4, 22.9e-11)
    tau_UV3 = E / dEdt_KN(E, 23209.0, 11.89e-11)

    tau_ISRF = 1. / (1. / tau_CMB + 1. / tau_IR + 1. / tau_opt + 1. / tau_UV1 + 1. / tau_UV2 + 1. / tau_UV3)

    ax.plot(E, tau_ISRF / MYR, color='tab:orange', lw=2.3, ls=':', zorder=10, label=f'full Klein-Nishina')
    tau_all = 1. / (1. / tau_B + 1. / tau_ISRF)
    ax.plot(E, tau_all / MYR, color='tab:blue', ls='--', lw=3.3, zorder=10, label=f'B + full Klein-Nishina')

    ax.vlines(1e3, 1e-10, 1e10, color='tab:gray', lw=2.3, ls='--', zorder=0)
    ax.hlines(1, 1e-10, 1e10, color='tab:gray', lw=2.3, ls='--', zorder=0)
    # Add legend and save figure
    ax.legend(fontsize=18)
    savefig(fig, 'energy_losses.pdf')

if __name__ == "__main__":
    plot_losses()



