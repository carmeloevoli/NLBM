import crdb
import numpy as np

def print_column_names(tab):
    for icol, col_name in enumerate(tab.dtype.fields):
        print('%2i' % icol, col_name)

def dump_datafile(quantity, energyType, expName, subExpName, filename, combo_level=0):
    print(f'search for {quantity} as a function of {energyType} measured by {expName}')
    
    tab = crdb.query(quantity, energy_type=energyType, combo_level=combo_level, energy_convert_level=2, exp_dates=expName)
 
    subExpNames = set(tab["sub_exp"])
    print('number of datasets found : ', len(subExpNames))
    print(subExpNames)

    adsCodes = set(tab["ads"])
    print(adsCodes)

    items = [i for i in range(len(tab["sub_exp"])) if tab["sub_exp"][i] == subExpName]
    print('number of data : ', len(items))
    assert(len(items) > 0)

    print(f'dump on {filename}')
    with open(filename, 'w') as f:
        f.write(f'#source: CRDB\n')
        f.write(f'#Quantity: {quantity}\n')
        f.write(f'#EnergyType: {energyType}\n')
        f.write(f'#Experiment: {expName}\n')
        f.write(f'#ads: {tab["ads"][items[0]]}\n')
        f.write(f'#E_mean - y - errSta_lo - errSta_up - errSys_lo - errSys_up\n')
        for eBin, value, errSta, errSys in zip(tab["e_bin"][items], tab["value"][items], tab["err_sta"][items], tab["err_sys"][items]):
            eMean = np.sqrt(eBin[0] * eBin[1])
            f.write(f'{eMean:10.5e} {value:10.5e} {errSta[0]:10.5e} {errSta[1]:10.5e} {errSys[0]:10.5e} {errSys[1]:10.5e}\n')
    f.close()
    print('')

if __name__== '__main__':
    dump_datafile('B/C', 'EKN', 'AMS02', 'AMS02 (2011/05-2016/05)', 'AMS-02_BC_Ekn.txt')
    dump_datafile('B/C', 'EKN', 'CALET', 'CALET (2015/10-2022/02)', 'CALET_BC_Ekn.txt')
    dump_datafile('B/C', 'EKN', 'DAMPE', 'DAMPE (2016/01-2021/12)', 'DAMPE_BC_Ekn.txt')
    dump_datafile('C/O', 'EKN', 'CALET', 'CALET (2015/10-2019/10)', 'CALET_CO_Ekn.txt')
    dump_datafile('C/O', 'R', 'AMS02', 'AMS02 (2011/05-2018/05)', 'AMS-02_CO_R.txt')
    dump_datafile('H', 'EK', 'AMS02', 'AMS02 (2011/05-2018/05)', 'AMS-02_H_Ek.txt')
    dump_datafile('H', 'EK', 'CALET', 'CALET (2015/10-2018/08)', 'CALET_H_Ek.txt')
    dump_datafile('H', 'EK', 'DAMPE', 'DAMPE (2016/01-2018/06)', 'DAMPE_H_Ek.txt')
    dump_datafile('He', 'EKN', 'AMS02', 'AMS02 (2011/05-2018/05)', 'AMS-02_He_Ekn.txt')
    dump_datafile('He', 'EKN', 'CALET', 'CALET (2015/10-2022/04)', 'CALET_He_Ekn.txt')
    dump_datafile('He', 'EKN', 'DAMPE', 'DAMPE (2016/01-2020/06)', 'DAMPE_He_Ekn.txt')
    dump_datafile('e+', 'EK', 'AMS02', 'AMS02 (2011/05-2018/05)', 'AMS-02_e+_Ek.txt')    
    dump_datafile('1H-bar', 'EK', 'AMS02', 'AMS02 (2011/05-2018/05)', 'AMS-02_pbar_Ek.txt')    
    dump_datafile('1H-bar/e+', 'R', 'AMS02', 'AMS02 (2011/05-2015/05)', 'AMS-02_pbar_e+_R.txt')