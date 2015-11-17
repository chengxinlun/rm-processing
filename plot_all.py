import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
        error_fit.append(error[i])
    wave_fit = np.array(wave_fit)
    flux_fit = np.array(flux_fit)
    error_fit = np.array(error_fit)
    return [wave_fit, flux_fit, error_fit]


os.mkdir("coadded")
target_list = os.listdir("raw")
for each_target in target_list:
    os.chdir("raw/"+each_target)
    wave = pickle.load(open("wave.pkl"))
    flux = pickle.load(open("flux.pkl"))
    ivar = pickle.load(open("ivar.pkl"))
    [wave, flux, ivar] = mask_points(wave, flux, ivar)
    fig1 = plt.figure()
    plt.plot(wave, flux)
    fig1.savefig("../../coadded/"+each_target+".jpg")
    plt.close()
    [wave, flux, ivar] = extract_fit_part(wave, flux, ivar, 4980, 5020)
    fig2 = plt.figure()
    plt.plot(wave, flux, marker='o')
    fig2.savefig("../../coadded/"+each_target+"-o3.jpg")
    plt.close()
    os.chdir("../../")
