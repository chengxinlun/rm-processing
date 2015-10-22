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


# Gaussian fit & local continuum for narrowlines
def single_line_fit(wave, flux, error, line):
    fig = plt.figure()
    single_line_fit_func = lambda x, a, x0, sig, k, b: a * \
        np.exp(-(x - x0)**2 / (2 * sig**2)) + k * x + b
    guess = [max(flux),
             float(line),
             np.std(flux),
             (flux[0] - flux[-1]) / (wave[0] - wave[-1]),
             (-flux[0] * wave[-1] + flux[-1] * wave[0]) / (wave[0] - wave[-1])]
    plt.plot(wave, flux)
    try:
        (line_fit_result,
         line_fit_extra) = curve_fit(single_line_fit_func,
                                     wave,
                                     flux,
                                     p0=guess,
                                     sigma=error,
                                     maxfev=10000)
        line_fit_error = np.sqrt(np.diag(line_fit_extra))
    except Exception:
        plt.close()
        raise SpectraException(
            "Line " +
            str(line) +
            "fit failed")
    if line_fit_result[0] < 0:
        plt.close()
        raise SpectraException(
            "Line " +
            str(line) +
            "not prominent, unable to fit")
    # For debug purpose only
    single_line_plot_func = lambda x: single_line_fit_func(
        x,
        line_fit_result[0],
        line_fit_result[1],
        line_fit_result[2],
        line_fit_result[3],
        line_fit_result[4])
    expected = list(map(single_line_plot_func, wave))
    plt.plot(wave, expected)
    rcs = chisquare(flux, expected)[0] / (len(flux) - 3)
    if rcs > 10.0:
        plt.close()
        raise SpectraException(
            "Line " +
            str(line) +
            "reduced chi-square too large" +
            str(rcs))
    return [line_fit_result, line_fit_error, fig]


def hbeta_complex_fit_func(p, fjac=None, x=None, y=None, err=None):
    gaussian = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    model = lambda x: gaussian(x,
                               p[0],
                               p[1],
                               p[2]) + gaussian(x,
                                                p[3],
                                                p[4],
                                                p[5]) + gaussian(x,
                                                                 p[6],
                                                                 p[7],
                                                                 p[8]) + gaussian(x,
                                                                                  p[9],
                                                                                  p[10],
                                                                                  p[11]) + p[12] * x + p[13]
    status = 0
    return [status, (y - model(x)) / err]

# Fit Hbeta, OIII 4959, OIII 5007 all together!!
# Well... This can work?


def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    hbeta_complex_fit_func = models.Gaussian1D(5.0, 4853.0, 20.0, bounds = {"amplitude": [0, 15.0], "mean": [4833,4863]}) + \
            models.Gaussian1D(7.0, 4863.0, 3.0, bounds = {"amplitude": [0, 15.0], "mean": [4853, 4883]}) + \
            models.Gaussian1D(5.0, 4883.0, 25.0, bounds = {"amplitude": [0, 15.0], "mean": [4863, 4930]}) + \
            models.Gaussian1D(2.0, 4930.0, 1.5, bounds = {"amplitude": [0, 15.0], "mean": [4883, 4959]}) + \
            models.Gaussian1D(5.0, 4959.0, 1.5, bounds = {"amplitude": [0, 15.0], "mean": [4940, 4970]}) + \
            models.Gaussian1D(20.0, 5007.0, 3.0, bounds = {"amplitude": [0,50.0]}) + \
            models.Linear1D((flux[0] - flux[-1]) / (wave[0] - wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0]) / (wave[0] - wave[-1]))
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(hbeta_complex_fit_func, wave, flux, weights=1 / error**2)

    # except Exception:
    #    plt.close()
    #    raise SpectraException("Line Hbeta fit failed")
    # if line_fit_result[0] < 0 or line_fit_result[
    #        1] < 0 or line_fit_result[2] < 0 or line_fit_result[3] < 0:
    #    plt.close()
    #    raise SpectraException("Line Hbeta not prominent, unable to fit")
    # For debug purpose only
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    #plt.show()
    rcs = chisquare(flux, expected)[0] / np.abs(len(flux) - 14)
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
        "line/" +
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
        "line-fig/" +
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


def main_process(sid, line_set):
    mjd_list = os.listdir("raw/" + str(sid))
    os.chdir("line")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../line-fig")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../")
    for each_mjd in mjd_list:
        print(str(sid) + "-" + str(each_mjd))
        [wave, flux, fluxerr] = read_raw_data(sid, each_mjd)
        [wave, flux, fluxerr] = mask_points(wave, flux, fluxerr)
        for each_line in line_set.keys():
            if check_line(wave, line_set[each_line][1]) == False:
                continue
            print("Found " + each_line + " in range")
            # Extract the part of data for fitting
            [wave_fit,
             flux_fit,
             fluxerr_fit] = extract_fit_part(wave,
                                             flux,
                                             fluxerr,
                                             line_set[each_line][0],
                                             line_set[each_line][2])
            # Fitting
            if each_line == "Hbeta":
                try:
                    [fit_res, fit_err, figure_line] = hbeta_complex_fit(
                        wave_fit, flux_fit, fluxerr_fit)
                except SpectraException as reason:
                    print(str(reason))
                    exception_logging(sid, each_mjd, each_line, reason)
                    continue
            else:
                try:
                    [fit_res, fit_err, figure_line] = single_line_fit(
                        wave_fit, flux_fit, fluxerr_fit, line_set[each_line][1])
                except SpectraException as reason:
                    print(str(reason))
                    exception_logging(sid, each_mjd, each_line, reason)
                    continue
            output_fig(figure_line, str(sid), str(each_mjd), each_line)
            os.chdir("line/" + str(sid))
            try:
                os.mkdir(each_line)
            except OSError:
                pass
            os.chdir("../../")
            output_fit(fit_res, fit_err, sid, each_mjd, each_line)
            print("Process finished for " + each_line + "\n")
        print("Process finished for " + each_mjd + "\n")


# line_set = {
#    "C4": [
#        1479.0, 1549.0, 1619.0], "Mg2": [
#            2745.0, 2798.0, 2858.0], "Hbeta": [
#                4720, 4902.0, 5200.0]}
line_set = {"Hbeta": [4720.0, 4902.0, 5150.0]}
try:
    os.mkdir("line")
except OSError:
    pass
try:
    os.mkdir("line-fig")
except OSError:
    pass
sid_list = get_total_sid_list()
# sid_list = ['160']
for each_sid in sid_list:
    try:
        main_process(str(each_sid), line_set)
    except Exception:
        pass
