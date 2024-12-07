import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Consolidate the common logic for loading and interpolating cross-section data
def load_and_interpolate(file_path, usecols, shape, scale_factor=1, method='linear'):
    try:
        # Load the data from the file
        data = np.loadtxt(file_path, usecols=usecols, unpack=True)
    except IOError as e:
        raise FileNotFoundError(f"Error loading file: {file_path}") from e
    
    # Extract relevant columns
    projectile_kinetic_energy = np.linspace(np.log10(shape[0][0]), np.log10(shape[0][1]), shape[0][2])
    secondary_kinetic_energy = np.linspace(np.log10(shape[1][0]), np.log10(shape[1][1]), shape[1][2])
    points = (projectile_kinetic_energy, secondary_kinetic_energy)

    # Reshape and scale cross-section data
    cross_sections = {}
    for name, table_data in zip(['pp', 'pHe', 'Hep', 'HeHe'], data[2:]):
        cross_section = table_data.reshape((len(projectile_kinetic_energy), len(secondary_kinetic_energy)))
        cross_sections[name] = RegularGridInterpolator(points, cross_section * scale_factor, method=method, bounds_error=False, fill_value=0.)

    return cross_sections

# Function for dsigmadE_ap - Proton/Antiproton cross-sections
def dsigmadE_ap(method='linear'):
    file_path = '../tables/supplementary__XS_table_Param_II_B.txt'
    usecols = (0, 1, 2, 3, 8, 9)
    shape = [(1., 1e7, 211), (0.1, 1e4, 151)]
    scale_factor = 1e31  # Convert m² to mbarn
    return load_and_interpolate(file_path, usecols, shape, scale_factor, method)

# Function for dsigmadE_pos - Positron cross-sections
def dsigmadE_pos(method='linear'):
    file_path = '../tables/supplementary_table_positrons_best_fit.txt'
    usecols = (0, 1, 2, 3, 8, 9)
    shape = [(0.1, 1e6, 140), (0.01, 1e4, 90)]
    scale_factor = 1  # No conversion needed
    return load_and_interpolate(file_path, usecols, shape, scale_factor, method)

# Plotting function
def plot_cross_sections(ax, x, Ep, interpolators, colors, linestyles):
    for (label, interpolator), color, linestyle in zip(interpolators.items(), colors, linestyles):
        s = [interpolator([np.log10(Ep), np.log10(x_i * Ep)])[0] for x_i in x]
        ax.plot(x, x * x * np.array(s) * Ep, color=color, linestyle=linestyle, label=label)

# Test function to plot the cross-sections
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
    
    # Define projectile energy and x range
    Ep = 1e3
    x = np.logspace(-4, 0, 1000)
    
    # Interpolate positron cross-sections
    pos_interpolators = dsigmadE_pos('linear')
    plot_cross_sections(
        ax, x, Ep, pos_interpolators,
        colors=['tab:orange', 'tab:red', 'tab:green', 'tab:purple'], linestyles=['-', '-', '-', '-']
    )

    # Interpolate antiproton cross-sections
    ap_interpolators = dsigmadE_ap('linear')
    plot_cross_sections(
        ax, x, Ep, ap_interpolators,
        colors=['tab:orange', 'tab:red', 'tab:green', 'tab:purple'], linestyles=[':', ':', ':', ':']
    )

    plt.legend()
    plt.savefig('antimatter_xsecs.pdf')

# Run the test function
if __name__ == "__main__":
    test_apxsec()
