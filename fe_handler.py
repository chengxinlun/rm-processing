import math
import os
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.pylab as plt
from astropy.modeling import models, fitting

# Define a special class for raising any exception related during the fit


class SpectraException(Exception):
    pass


# Recommended method to read a single pkl file
def read_database(filein):
    readfile = open(filein, "rb")
    data = pickle.load(readfile)
    readfile.close()
    return data


# Recommended method to read raw data
def read_raw_data(sid, mjd):
    os.chdir("raw/" + str(sid) + "/" + str(mjd))
    wave = read_database("wave.pkl")
    flux = read_database("flux.pkl")
    fluxerr = read_database("fluxerr.pkl")
    os.chdir("../../../")
    return [wave, flux, fluxerr]


# Check whether wavelength @wave_line is inside list of wave
def check_line(wave, wave_line):
    max_wave = max(wave)
    min_wave = min(wave)
    if ((max_wave > (wave_line + 150.0))
            and (min_wave < (wave_line - 150.0))) == True:
        return True
    else:
        return False


# Get the whole sid list
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


# Omit error=0 points
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


# Function to extract part of the spectra to fit
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


def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (FeII, Hgamma, OIII), (FeII, FeII, FeII), FeII, (Hbeta, FeII, OIII, OIII)
    hbeta_complex_fit_func = models.Gaussian1D(2.0, 4318.0, 1.0, bounds = {"amplitude": [0, 10.0], "mean": [4300, 4340]}) + \
            models.Gaussian1D(9.0, 4346.0, 10.0, bounds = {"amplitude": [0, 50.0], "mean": [4300, 4360]}) + \
            models.Gaussian1D(5.0, 4522.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4510, 4530]}) + \
            models.Gaussian1D(5.0, 4549.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4540, 4555]}) + \
            models.Gaussian1D(5.0, 4583.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4575, 4590]}) + \
            models.Gaussian1D(2.0, 4629.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4620, 4650]}) + \
            models.Gaussian1D(2.0, 4660.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4650, 4670]}) + \
            models.Lorentz1D(20.0, 4853.0, 20.0, bounds = {"amplitude": [0, 50.0], "x_0": [4843, 4873]}) + \
            models.Gaussian1D(2.0, 4930.0, 1.5, bounds = {"amplitude": [0, 50.0], "mean": [4883, 4959]}) + \
            models.Gaussian1D(5.0, 4959.0, 1.5, bounds = {"amplitude": [0, 50.0], "mean": [4940, 4970]}) + \
            models.Gaussian1D(10.0, 5007.0, 3.0, bounds = {"amplitude": [0,50.0]}) + \
            models.Gaussian1D(5.0, 5169.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5150, 5180]}) + \
            models.Gaussian1D(5.0, 5197.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5180, 5210]}) + \
            models.Gaussian1D(2.0, 5234.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5220, 5250]}) + \
            models.Gaussian1D(2.0, 5276.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5260, 5300]}) + \
            models.Gaussian1D(5.0, 5316.0, 2.0, bounds = {"amplitude": [0, 10.0], "mean": [5300, 5325]}) + \
            models.PowerLaw1D(flux[0], wave[0], -np.log(min(flux)/flux[0])/np.log(wave[flux.argmin()]/wave[0]), fixed = {"x_0": True})
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(hbeta_complex_fit_func, wave, flux, weights=1 / error**2)
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    plt.show()
    print(fit.parameters)
    rcs = chisquare(flux, expected)[0] / np.abs(len(flux) - 50)
    print(rcs)
    if rcs > 10.0:
        plt.close()
        raise SpectraException(
            "Line Hbeta reduced chi-square too large" +
            str(rcs))
    line_fit_error = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return [fit.parameters, line_fit_error, fig]


# Function to output fit result and error
def output_fit(fit_result, fit_error, sid, mjd, line):
    picklefile = open(
        "Fe2/" +
        str(sid) +
        "/" +
        str(line) +
        "/" +
        str(mjd) +
        ".pkl",
        "wb")
    pickle.dump(fit_result, picklefile)
    pickle.dump(fit_error, picklefile)
    picklefile.close()


# Function to output figure
def output_fig(fig, sid, mjd, line):
    fig.savefig(
        "Fe2-fig/" +
        str(sid) +
        "/" +
        str(mjd) +
        "-" +
        str(line) +
        ".jpg")
    plt.close()


# Exception logging process
def exception_logging(sid, mjd, line, reason):
    log = open("fit_error.log", "a")
    log.write(
        str(sid) +
        " " +
        str(mjd) +
        " " +
        str(line) +
        " " +
        str(reason) +
        "\n")
    log.close()


def main_process(sid, line):
    mjd_list = os.listdir("raw/" + str(sid))
    os.chdir("Fe2")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../Fe2-fig")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../")
    for each_mjd in mjd_list:
        print(str(sid) + "-" + str(each_mjd))
        [wave, flux, fluxerr] = read_raw_data(sid, each_mjd)
        [wave, flux, fluxerr] = mask_points(wave, flux, fluxerr)
        # Extract the part of data for fitting
        [wave_fit,
         flux_fit,
         fluxerr_fit] = extract_fit_part(wave, flux, fluxerr, line[0], line[2])
        #try:
        [fit_res, fit_err, figure_line] = hbeta_complex_fit(
            wave_fit, flux_fit, fluxerr_fit)
        #except SpectraException as reason:
        #    print(str(reason))
        #    exception_logging(sid, each_mjd, "Fe2", reason)
        #    continue
        output_fig(figure_line, str(sid), str(each_mjd), "Fe2")
        os.chdir("Fe2/" + str(sid))
        try:
            os.mkdir("Fe2")
        except OSError:
            pass
        os.chdir("../../")
        output_fit(fit_res, fit_err, sid, each_mjd, "Fe2")
        print("Process finished for " + each_mjd + "\n")


line = [4250.0, 4902.0, 5400.0]
try:
    os.mkdir("Fe2")
except OSError:
    pass
try:
    os.mkdir("Fe2-fig")
except OSError:
    pass
sid_list = [316]
for each_sid in sid_list:
    #try:
    main_process(str(each_sid), line)
    #except Exception:
    #    pass
