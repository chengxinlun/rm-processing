import math
import os
import numpy as np
import pickle
from scipy.integrate import quad
from scipy.stats import chisquare
import matplotlib.pylab as plt
from astropy.modeling import models, fitting
import warnings

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
def read_raw_data(sid):
    os.chdir("raw/" + str(sid))
    wave = read_database("wave.pkl")
    flux = read_database("flux.pkl")
    fluxerr = read_database("ivar.pkl")
    os.chdir("../../")
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


# Function to fit Hbeta
def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (Hbeta, FeII, OIII, OIII, FeII)
    hbeta_complex_fit_func = models.Lorentz1D(20.0, 4853.0, 40.0, bounds = {"amplitude": [0, 50.0], "x_0": [4833,4873]}) + \
            models.Gaussian1D(5.0, 4930.0, 4.0, bounds = {"amplitude": [0, 20.0], "mean": [4900, 4940]}) + \
            models.Gaussian1D(10.0, 4960.0, 4.0, bounds = {"amplitude": [0, 50.0], "mean": [4955, 4980]}) + \
            models.Gaussian1D(20.0, 5007.0, 6.0, bounds = {"amplitude": [0, 50.0]}) + \
            models.Gaussian1D(5.0, 5018.0, 4.0, bounds = {"amplitude": [0, 20.0], "mean": [5007, 5050]}) + \
            models.Linear1D((flux[0] - flux[-1])/(wave[0]-wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0])/(wave[0]-wave[-1]))
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 10000)
        except Warning:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.Linear1D(fit.parameters[15], fit.parameters[16])
            plt.plot(wave, cont(wave))
            fig.savefig("Hbeta-failed")
            plt.close()
            raise SpectraException("Line Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[15], fit.parameters[16])
    plt.plot(wave, cont(wave))
    fig.savefig("Hbeta.png")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters


# Function to fit FeII lines before Hbeta
def fe2_before_hbeta(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (Hgamma, OIII), (FeII, FeII, FeII, FeII)
    hbeta_complex_fit_func = models.Gaussian1D(9.0, 4346.0, 10.0, bounds = {"amplitude": [0, 50.0], "mean": [4300, 4360]}) + \
            models.Gaussian1D(5.0, 4522.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4510, 4530]}) + \
            models.Gaussian1D(5.0, 4549.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4540, 4555]}) + \
            models.Gaussian1D(5.0, 4583.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4575, 4590]}) + \
            models.Gaussian1D(2.0, 4629.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4620, 4650]}) + \
            models.Gaussian1D(2.0, 4660.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4650, 4670]}) + \
            models.Linear1D((flux[0] - flux[-1])/(wave[0]-wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0])/(wave[0]-wave[-1]))
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights= error, maxiter = 10000)
        except Warning:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
            plt.plot(wave, cont(wave))
            fig.savefig("bef-failed")
            plt.close()
            raise SpectraException("Line Fe2 before Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
    plt.plot(wave, cont(wave))
    fig.savefig("bef")
    plt.close()
    rcs = chisquare(flux, expected)[0] / np.abs(len(flux) - 20)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Fe2 before Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters


# Function to fit FeII lines after Hbeta
def fe2_after_hbeta(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (FeII, FeII), (FeII, FeII), FeII
    hbeta_complex_fit_func = models.Gaussian1D(5.0, 5169.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5150, 5180]}) + \
            models.Gaussian1D(5.0, 5197.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5180, 5210]}) + \
            models.Gaussian1D(2.0, 5234.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5220, 5250]}) + \
            models.Gaussian1D(2.0, 5276.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5260, 5300]}) + \
            models.Gaussian1D(5.0, 5316.0, 2.0, bounds = {"amplitude": [0, 10.0], "mean": [5300, 5325]}) + \
            models.Linear1D((flux[0] - flux[-1])/(wave[0]-wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0])/(wave[0]-wave[-1]))
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights= error, maxiter = 10000)
        except Warning:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.Linear1D(fit.parameters[15], fit.parameters[16])
            plt.plot(wave, cont(wave))
            fig.savefig("aft-failed")
            plt.close()
            raise SpectraException("Line Fe2 after Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[15], fit.parameters[16])
    plt.plot(wave, cont(wave))
    fig.savefig("aft")
    plt.close()
    rcs = chisquare(flux, expected)[0] / np.abs(len(flux) - 17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Fe2 after Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters


# Function to find union of ranges
def union(a):
    b = []
    for begin,end in sorted(a):
        if b and b[-1][1] >= begin - 1:
            b[-1] = (b[-1][0], end)
        else:
            b.append((begin, end))
    return b


# Compare OIII and Fe2
def compare_fe2(wave, flux, error):
    # First fit Hbeta and OIII
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 4720.0, 5100.0)
    hbeta_res = hbeta_complex_fit(wave_fit, flux_fit, error_fit)
    o3range = [hbeta_res[10] - 3.0 * hbeta_res[11], hbeta_res[10] + 3.0 * hbeta_res[11]]
    o3cont = lambda x: hbeta_res[15] * x + hbeta_res[16]
    o3flux = hbeta_res[9]
    o3contflux = o3cont(hbeta_res[10])
    o3sn = o3flux/o3contflux
    # Fit FeII lines before Hbeta complex
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 4270.0, 4720.0)
    bef_res = fe2_before_hbeta(wave_fit, flux_fit, error_fit)
    # Fit FeII lines after Hbeta complex
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 5120.0, 5400.0)
    aft_res = fe2_after_hbeta(wave_fit, flux_fit, error_fit)
    # Get total flux of each Fe line
    fecont = lambda x: bef_res[18] * x + bef_res[19]
    feflux = bef_res[0] + bef_res[3] + bef_res[6] + bef_res[9] + bef_res[12] + bef_res[15]
    fecontflux = fecont(bef_res[1]) + fecont(bef_res[4]) + fecont(bef_res[10]) + fecont(bef_res[13]) + fecont(bef_res[16])
    fecont = lambda x: aft_res[15] * x + aft_res[16]
    feflux = feflux + aft_res[0] + aft_res[3] + aft_res[6] + aft_res[9] + aft_res[12]
    fecontflux = fecontflux + fecont(aft_res[1]) + fecont(aft_res[4]) + fecont(aft_res[7]) + fecont(aft_res[10]) + fecont(aft_res[13])
    fesn = feflux / fecontflux
    return [hbeta_res, bef_res, aft_res, o3sn, fesn]


# Function to output fit result and error
def output_fit(fit_result, sid, line):
    picklefile = open(
        "Fe2/" +
        str(sid) +
        "/" +
        str(line) +
        ".pkl",
        "wb")
    pickle.dump(fit_result, picklefile)
    picklefile.close()


# Exception logging process
def exception_logging(sid, line, reason):
    log = open("Fe2_fit_error.log", "a")
    log.write(
        str(sid) +
        " " +
        str(line) +
        " " +
        str(reason) +
        "\n")
    log.close()


def main_process(sid, line):
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
    [wave, flux, fluxerr] = read_raw_data(sid)
    [wave, flux, fluxerr] = mask_points(wave, flux, fluxerr)
    # Extract the part of data for fitting
    [wave, flux, fluxerr] = extract_fit_part(wave, flux, fluxerr, line[0], line[2])
    os.chdir("Fe2-fig/" + str(sid))
    try:
        [hbeta, bef, aft, o3sn, fesn] = compare_fe2(wave, flux, fluxerr)
    except Exception as reason:
        print(str(reason))
        exception_logging(sid, "Fe2", reason)
        os.chdir("../../")
        return
    os.chdir("../../")
    output_fit(hbeta, sid, "Hbeta")
    output_fit(bef, sid, "bef")
    output_fit(aft, sid, "aft")
    os.chdir("Fe2/")
    sn_file = open(str(sid) + ".txt", "a")
    sn_file.write("%9.4f    %9.4f\n" % (o3sn, fesn))
    sn_file.close()
    os.chdir("../")


line = [4250.0, 4902.0, 5400.0]
try:
    os.mkdir("Fe2")
except OSError:
    pass
try:
    os.mkdir("Fe2-fig")
except OSError:
    pass
sid_list = get_total_sid_list()
#sid_list = [171]
for each_sid in sid_list:
    #try:
    main_process(str(each_sid), line)
    #except Exception:
    #    pass
