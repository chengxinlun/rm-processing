import math
import os
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.pylab as plt


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
        raise SpectraException(
            "Line " +
            str(line) +
            "fit failed")
    if line_fit_result[0] < 0:
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
        raise SpectraException(
            "Line " +
            str(line) +
            "reduced chi-square too large" +
            str(rcs))
    return [line_fit_result, line_fit_error, fig]


# Fit Hbeta, OIII 4959, OIII 5007 all together!!
# Well... This can work?
def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    gaussian = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    hbeta_complex_fit_func = lambda x, a1, a2, a3, a4, x01, x02, x03, x04, sig1, sig2, sig3, sig4, k, b: gaussian(
        x, a1, x01, sig1) + gaussian(x, a2, x02, sig2) + gaussian(x, a3, x03, sig3) + gaussian(x, a4, x04, sig4) + k * x + b
    plt.plot(wave, flux)
    guess = [1.0,
             1.0,
             1.0,
             2.0,
             4902.0,
             4902.0,
             5007.0,
             4959.0,
             (flux[0] - flux[-1]) / (wave[0] - wave[-1]),
             (-flux[0] * wave[-1] + flux[-1] * wave[0]) / (wave[0] - wave[-1])]
    try:
        (line_fit_result,
         line_fit_extra) = curve_fit(hbeta_complex_fit_func,
                                     wave,
                                     flux,
                                     p0=guess,
                                     sigma=error,
                                     maxfev=100000)
        line_fit_error = np.sqrt(np.diag(line_fit_extra))
    except Exception:
        plt.close()
        raise SpectraException("Line Hbeta fit failed")
    if line_fit_result[0] < 0 or line_fit_result[
            1] < 0 or line_fit_result[2] < 0 or line_fit_result[3] < 0:
        plt.close()
        raise SpectraException("Line Hbeta not prominent, unable to fit")
    # For debug purpose only
    hbeta_complex_plot_func = lambda x: hbeta_complex_fit_func(
        x,
        line_fit_result[0],
        line_fit_result[1],
        line_fit_result[2],
        line_fit_result[3],
        line_fit_result[4],
        line_fit_result[5],
        line_fit_result[6],
        line_fit_result[7],
        line_fit_result[8],
        line_fit_result[9],
        line_fit_result[10],
        line_fit_result[11])
    expected = list(map(hbeta_line_complex_func, wave))
    plt.plot(wave, expected)
    rcs = chisquare(flux, expected) / (len(flux) - 12)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line " + str(line) + "reduced chi-square too large" + str(rcs))
    return [line_fit_result, line_fit_error, figure]


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


line_set = {
    "C4": [
        1479.0, 1549.0, 1619.0], "Mg2": [
            2745.0, 2798.0, 2858.0], "Hbeta": [
                4720, 4902.0, 5200.0]}
try:
    os.mkdir("line")
except OSError:
    pass
try:
    os.mkdir("line-fig")
except OSError:
    pass
sid_list = get_total_sid_list()
for each_sid in sid_list:
    try:
        main_process(str(each_sid), line_set)
    except Exception:
        pass
