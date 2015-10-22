import math
import os
import numpy as np
import pickle
from psrm.base.target_fibermap import parseVar_sid
import matplotlib.pylab as plt


def read_database(filein):
    readfile = open(filein, "rb")
    data = pickle.load(readfile)
    readfile.close()
    return data

def read_raw_data(sid, mjd):
    os.chdir("raw/" + str(sid) + "/" + str(mjd))
    wave = read_database("wave.pkl")
    flux = read_database("flux.pkl")
    fluxerr = read_database("fluxerr.pkl")
    os.chdir("../../../")
    return [wave, flux, fluxerr]

def check_line(wave, wave_line):
    max_wave = max(wave)
    min_wave = min(wave)
    if ((max_wave > (wave_line + 150.0)) and (min_wave < (wave_line - 150.0))) == True:
        return True
    else:
        return False


def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


def mask_points(wave, flux, error):
    wave_temp = list()
    flux_temp = list()
    error_temp = list()
    for i in range(len(error)):
        if error[i] == 0:
            continue
        wave_temp.append(wave[i])
        flux_temp.append(flux[i])
        error_temp.append(error[i])
    return [wave_temp, flux_temp, error_temp]


def extract_fit_part(wave, flux, error, min_wave, max_wave):
    wave_fit = list()
    flux_fit = list()
    error_fit = list()
    for i in range(len(wave)):
        if wave[i] < min_wave:
            continue
        if wave[i] > max_wave:
            break
        wave_fit.append(wave[i])
        flux_fit.append(flux[i])
        error_fit.append(error[i] / flux[i])
    wave_fit = np.array(wave_fit)
    flux_fit = np.array(flux_fit)
    error_fit = np.array(error_fit)
    return [wave_fit, flux_fit, error_fit]

def main_process():
    sid_list = get_total_sid_list()
    for sid in sid_list:
        try:
            [wave, flux, error] = read_raw_data(str(sid), 56660)
            [wave, flux, error] = mask_points(wave, flux, error)
            if check_line(wave, 5800.0):
                info = parseVar_sid(sid, 'indx')
                rmid = info[sid]['indx']
                [wave, flux, error] = extract_fit_part(wave, flux, error, 4150.0,5800.0)
                fig = plt.figure()
                plt.plot(wave, flux)
                fig.savefig(str(rmid)+".jpg")
                plt.close()
        except Exception:
            continue


main_process()
