from psrm.base.target_fibermap import parseVar_sid
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import pickle


def z2dl(z):
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    dl_MPC = cosmo.luminosity_distance(z)
    dl_cm = dl_MPC.value * 3.085677581 * (10.0 ** 24.0)
    return float(dl_cm)


def luminosity(dl_cm, flux):
    return 4.0 * 3.1415926 * dl_cm * dl_cm * flux * 10. ** (0. - 17.)


def lag_corr(z, lag):
    return lag / (1.0 + z)


def get_z():
    zfinal_file = open("info_database/zfinal.pkl", "rb")
    zfinal_dict = pickle.load(zfinal_file)
    zfinal_file.close()
    return zfinal_dict


# Get the whole sid list
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


# Recommended function to read lag and continuum
def read_data_non_pkl(filein):
    infile = open(filein)
    data_list = list()
    error_list = list()
    sid_list = list()
    for each in infile:
        data = data_list.append(float(each.split("    ")[0]))
        error = error_list.append(float(each.split("    ")[1]))
        sid = sid_list.append(float(each.split("    ")[2]))
    return [data_list, error_list, sid_list]


def main_process(line, zfinal_dict):
    [lag, lag_err, sid_lag] = read_data_non_pkl("lag/" + line + ".txt")
    [cont, cont_err, sid_cont] = read_data_non_pkl("lag/" + line + "-cont.txt")
    lum = list()
    lum_err = list()
    lag_corrected = list()
    lag_corrected_err = list()
    for i in range(len(lag)):
        zfinal = zfinal_dict[sid_lag[i]]
        dl_cm = z2dl(zfinal)
        lum.append(luminosity(dl_cm, cont[i]))
        lum_err.append(luminosity(dl_cm, cont_err[i]))
        lag_corrected.append(lag_corr(zfinal, lag[i]))
        lag_corrected_err.append(lag_corr(zfinal, lag_err[i]))
    output = open("corrected/" + line + ".pkl", "wb")
    pickle.dump(lum, output)
    pickle.dump(lum_err, output)
    pickle.dump(lag_corrected, output)
    pickle.dump(lag_corrected_err, output)
    output.close()


line_set = ["Hbeta", "C4", "N5", "Mg2"]
try:
    os.mkdir("corrected")
except OSError:
    pass
zfinal_dict = get_z()
for each_line in line_set:
    main_process(each_line, zfinal_dict)
