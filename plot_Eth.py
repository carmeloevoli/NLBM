import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

mp = 0.938 # GeV

def compute_Eth(E):
    mp_2 = mp * mp
    mp_3 = mp_2 * mp
    mp_4 = mp_3 * mp
    a = 2. * (mp * E - mp_2)
    b = -(2. * mp * E * E + 4. * mp_2 * E - 6. * mp_3)
    c = -(2. * mp_2 * E * E + 6. * mp_3 * E + 8. * mp_4)
    return (-b + np.sqrt(b * b - 4. * a * c)) / 2. / a

def plot_Eth():
    def set_axes(ax):
        ax.set_xlabel(r'T$_{\bar p}$ [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([0.1, 1e2])
        ax.set_ylabel(r'T$_{\rm th}$ [GeV]')
        ax.set_yscale('log')
        ax.set_ylim([4, 1e2])

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    T = np.logspace(-1, 2, 1000)
    E_th = compute_Eth(T + mp)
    
    ax.plot(T, E_th - mp)
    ax.plot(T, T, ':')

    ax.hlines(6. * mp, .1, 1e2, ls=':', lw=2, color='tab:gray')
    ax.text(50, 6.1, r'6 m$_{\rm p}$', fontsize=21)

    ax.legend(fontsize=17, loc='lower left')
    plt.savefig('NLBM_Eth.pdf')

if __name__== "__main__":
    plot_Eth()
