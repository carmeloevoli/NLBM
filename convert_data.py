import numpy as np

def get_data(filename, minEnergy, maxEnergy = 1e20):
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    err_tot_lo = np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    err_tot_up = np.sqrt(err_stat_up**2. + err_sys_up**2.)
    items = [i for i in range(len(E)) if (E[i] > minEnergy and E[i] < maxEnergy)]
    return [E[items], y[items], err_tot_lo[items], err_tot_up[items]]
    
def dump_data(filename, data):
    f = open('data/' + filename, 'w')
    x, y, err_tot_lo, err_tot_up = data
    for x_i, y_i, err_lo_i, err_up_i in zip(x, y, err_tot_lo, err_tot_up):
        f.write(f'{x_i:e} {y_i:e} {err_lo_i:e} {err_up_i:e}\n')
    f.close()
    
def convert_data():
    data = get_data('kiss_tables/AMS-02_B_C_kineticEnergyPerNucleon.txt', 10.)
    dump_data('AMS-02_BC.txt', data)
    
    data = get_data('kiss_tables/DAMPE_B_C_kineticEnergyPerNucleon.txt', 10.)
    dump_data('DAMPE_BC.txt', data)

    data = get_data('kiss_tables/CALET_B_C_kineticEnergyPerNucleon.txt', 10.)
    dump_data('CALET_BC.txt', data)
    
    data = get_data('kiss_tables/AMS-02_C_O_rigidity.txt', 10.)
    data[0] /= 2. # R -> Ekn
    dump_data('AMS-02_CO.txt', data)
    
    data = get_data('kiss_tables/CALET_C_O_kineticEnergyPerNucleon.txt', 10.)
    dump_data('CALET_CO.txt', data)

    data = get_data('kiss_tables/AMS-02_H_rigidity.txt', 10.)
    dump_data('AMS-02_H.txt', data)
    
    data = get_data('kiss_tables/CALET_H_kineticEnergy.txt', 10.)
    dump_data('CALET_H.txt', data)

    data = get_data('kiss_tables/DAMPE_H_totalEnergy.txt', 10.)
    dump_data('DAMPE_H.txt', data)

    data = get_data('kiss_tables/AMS-02_He_rigidity.txt', 10.)
    data[0] /= 2. # R -> Ekn
    data[1] *= 2.
    data[2] *= 2.
    data[3] *= 2.
    dump_data('AMS-02_He.txt', data)

    data = get_data('kiss_tables/CALET_He_kineticEnergy.txt', 10.)
    data[0] /= 4. # R -> Ekn
    data[1] *= 4.
    data[2] *= 4.
    data[3] *= 4.
    dump_data('CALET_He.txt', data)

    data = get_data('kiss_tables/DAMPE_He_totalEnergy.txt', 10.)
    data[0] /= 4. # R -> Ekn
    data[1] *= 4.
    data[2] *= 4.
    data[3] *= 4.
    dump_data('DAMPE_He.txt', data)
    
if __name__== "__main__":
    convert_data()
