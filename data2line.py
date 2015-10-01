import math
import os
import numpy as np
import pickle
from scipy.optimize import curve_fit
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


# Omit error=0 or flux<0 points
def mask_points(wave, flux, error):
    wave_temp = list()
    flux_temp = list()
    error_temp = list()
    for i in range(len(error)):
        if error[i] == 0 or flux[i] < 0:
            continue
        wave_temp.append(wave[i])
        flux_temp.append(flux[i])
        error_temp.append(error[i])
    return [wave_temp, flux_temp, error_temp]


# Function to extract part of the spectra to fit
def extract_fit_part(wave, flux, error, center, width):
    wave_fit = list()
    flux_fit = list()
    error_fit = list()
    for i in range(len(wave)):
        if wave[i] < (center - width):
            continue
        if wave[i] > (center + width):
            break
        wave_fit.append(wave[i])
        flux_fit.append(flux[i])
        error_fit.append(error[i] / flux[i])
    return [wave_fit, flux_fit, error_fit]


# Fit continuum with nearby data
def fit_cont(wave, flux, error, con_wave):
    fig = plt.figure()
    wave_fit = list()
    flux_fit = list()
    error_fit = list()
    for each_part in con_wave:
        [wave_temp, flux_temp, error_temp] = extract_fit_part(
            wave, flux, error, each_part, 10.0)
        wave_fit.extend(wave_temp)
        flux_fit.extend(flux_temp)
        error_fit.extend(error_temp)
    powerlaw = lambda x, a, b: a * (x ** b)
    guess = [10.0, -1.0]
    plt.plot(wave_fit, flux_fit)
    try:
        times = 0
        while times < 3:
            (con_fit_result, con_fit_extra) = curve_fit(powerlaw,
                                                    wave_fit,
                                                    flux_fit,
                                                    p0=guess,
                                                    sigma=error_fit,
                                                    maxfev=100000)
            print(con_fit_result)
            guess = con_fit_result
            times = times + 1
        powerlaw_plot_func = lambda x: powerlaw(
            x,
            con_fit_result[0],
            con_fit_result[1])
        plt.plot(wave_fit, list(map(powerlaw_plot_func, wave_fit)))
        con_fit_error = np.sqrt(np.diag(con_fit_extra))
    except Exception as reason:
        raise SpectraException("Fit continuum failed")
    return [con_fit_result, con_fit_error, wave_fit, flux_fit, error_fit, fig]


# Continuum correction
def corr_cont(wave, flux, err, cont_res):
    cont_powerlaw = lambda x: cont_res[0] * (x ** cont_res[1])
    flux_corr = list()
    for i in range(len(wave)):
        flux_corr.append(flux[i] - cont_powerlaw(wave[i]))
    return flux_corr


# Gaussian fit for narrowlines
def single_line_fit(wave, flux, error, line):
    fig = plt.figure()
    gaussian = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    guess = [max(flux), float(line), np.std(flux)]
    plt.plot(wave, flux)
    try:
        (line_fit_result,
         line_fit_extra) = curve_fit(gaussian,
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
            "not prominent, unable to fit")
    if line_fit_result[0] < 0:
        raise SpectraException(
            "Line " +
            str(line) +
            "not prominent, unable to fit")
    # For debug purpose only
    single_line_plot_func = lambda x: gaussian(
       x,
       line_fit_result[0],
       line_fit_result[1],
       line_fit_result[2])
    plt.plot(wave, list(map(single_line_plot_func, wave)))
    return [line_fit_result, line_fit_error, fig]


# Fit Hbeta
def hbeta_complex_fit(wave_fit, flux_fit, error_fit):
    fig = plt.figure()
    gaussian = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    #plt.plot(wave_fit, flux_fit)
    # First fit OIII 5007 and substract it
    guess_o3la = [5.0, 5007.0, 10.0]
    try:
        (o3laline_fit_result,
         o3laline_fit_extra) = curve_fit(gaussian,
                                         wave_fit,
                                         flux_fit,
                                         p0=guess_o3la,
                                         sigma=error_fit,
                                         maxfev=100000)
    except Exception:
        raise SpectraException("Line Hbeta fit failed at OIII 5007")
    if o3laline_fit_result[0] < 0:
        raise SpectraException("Line Hbeta fit failed at OIII 5007")
    for i in range(len(flux_fit)):
        flux_fit[i] = flux_fit[i] - gaussian(wave_fit[i],
                                             o3laline_fit_result[0],
                                             o3laline_fit_result[1],
                                             o3laline_fit_result[2])
    # Second, fit Hbeta 4902 and substract it
    guess_hbetanarr = [4.0, 4902.0, 10.0]
    try:
        (hbetanarr_fit_result,
         hbetanarr_fit_extra) = curve_fit(gaussian,
                                          wave_fit,
                                          flux_fit,
                                          p0=guess_hbetanarr,
                                          sigma=error_fit,
                                          maxfev=10000)
    except Exception:
        raise SpectraException("Line Hbeta fit failed at Hbeta 4902")
    if hbetanarr_fit_result[0] < 0:
        raise SpectraException("Line Hbeta fit failed at Hbeta 4902")
    for i in range(len(wave_fit)):
        flux_fit[i] = flux_fit[i] - gaussian(wave_fit[i],
                                             hbetanarr_fit_result[0],
                                             hbetanarr_fit_result[1],
                                             hbetanarr_fit_result[2])
    # Third, fit the rest
    guess_hbeta = [3.0, 4959.0, 10.0]
    try:
        (hbetaline_fit_result,
         hbetaline_fit_extra) = curve_fit(gaussian,
                                          wave_fit,
                                          flux_fit,
                                          p0=guess_hbeta,
                                          sigma=error_fit,
                                          maxfev=10000)
    except Exception:
        raise SpectraException("Line Hbeta fit failed at OIII 4959")
    # Merge the result for output
    line_fit_result_temp = list()
    line_fit_extra_temp = list()
    for each in o3laline_fit_result:
        line_fit_result_temp.append(each)
    for each in hbetanarr_fit_result:
        line_fit_result_temp.append(each)
    for each in hbetaline_fit_result:
        line_fit_result_temp.append(each)
    line_fit_result = np.array(line_fit_result_temp)
    try:
        o3laline_fit_error = np.sqrt(np.diag(o3laline_fit_extra))
        hbetanarr_fit_error = np.sqrt(np.diag(hbetanarr_fit_extra))
        hbetaline_fit_error = np.sqrt(np.diag(hbetaline_fit_extra))
    except Exception:
        raise SpectraException(
            "Line Hbeta fit failed becaused of unphysical error")
    for each in o3laline_fit_error:
        line_fit_extra_temp.append(each)
    for each in hbetanarr_fit_error:
        line_fit_extra_temp.append(each)
    for each in hbetaline_fit_error:
        line_fit_extra_temp.append(each)
    line_fit_error = np.array(line_fit_extra_temp)
    # For debug purpose only
    hbeta_line_plot_func = lambda x: gaussian(x,
                                              line_fit_result[0],
                                              line_fit_result[1],
                                              line_fit_result[2]) + gaussian(x,
                                                                             line_fit_result[3],
                                                                             line_fit_result[4],
                                                                             line_fit_result[5]) + gaussian(x,
                                                                                                            line_fit_result[6],
                                                                                                            line_fit_result[7],
                                                                                                            line_fit_result[8])
    plt.plot(wave_fit, list(map(hbeta_line_plot_func, wave_fit)))
    return [line_fit_result, line_fit_error, figure]


# Check whether fit is successful by reduced chi-square (If < 10.0, then fine)
def check_fit(wave, flux, error, fit_res, line):
    gaussian = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    kk_func = lambda x, y, z: ((x - y) / z) ** 2.0
    if line == "cont":
        powerlaw_func = lambda x: fit_res[0] * (x ** fit_res[1])
        expected = np.array(list(map(powerlaw_func, wave)))
    elif line != "Hbeta":
        narrline_func = lambda x: gaussian(
            x,
            fit_res[0],
            fit_res[1],
            fit_res[2])
        expected = np.array(list(map(narrline_func, wave)))
    else:
        hbeta_line_func = lambda x: gaussian(x,
                                             fit_res[0],
                                             fit_res[1],
                                             fit_res[2]) + gaussian(x,
                                                                    fit_res[3],
                                                                    fit_res[4],
                                                                    fit_res[5]) + gaussian(x,
                                                                                           fit_res[6],
                                                                                           fit_res[7],
                                                                                           fit_res[8])
        expected = np.array(list(map(hbeta_line_func, wave)))
    rkk = sum(list(map(kk_func, flux, expected, error))) / (len(flux) - 3.0)
    if rkk > 10.0:
        raise SpectraException(
            str(line) +
            " reduced chi-square (" +
            str(rkk) +
            ") too large")
    return rkk


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


def main_process(sid, line_set, cont_set):
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
            if check_line(wave, line_set[each_line]) == False:
                continue
            print("Found " + each_line + " in range")
            # Extract the part of data for fitting
            [wave_fit, flux_fit, fluxerr_fit] = extract_fit_part(
                    wave, flux, fluxerr, np.mean(cont_set[each_line]), 0.5 * (cont_set[each_line][1] - cont_set[each_line][0]))
            # Fit local continuum
            try:
                [cont_res,
                 cont_err,
                 wave_cont,
                 flux_cont,
                 fluxerr_cont,
                 figure_cont] = fit_cont(wave_fit,
                                          flux_fit,
                                          fluxerr_fit,
                                          cont_set[each_line])
            except SpectraException as reason:
                print(str(reason))
                exception_logging(sid, each_mjd, each_line, reason)
                continue
            figure_cont.savefig("line-fig/"+str(sid)+"/"+str(each_mjd)+"-"+str(each_line)+"-cont.jpg")
            plt.close()
            try:
                rkk = check_fit(
                    wave_cont,
                    flux_cont,
                    fluxerr_cont,
                    cont_res,
                    "cont")
            except SpectraException as reason:
                print(str(reason))
                exception_logging(sid, each_mjd, each_line + "-cont", reason)
                continue
            os.chdir("line/" + str(sid))
            try:
                os.mkdir(each_line + "-cont")
            except OSError:
                pass
            os.chdir("../../")
            output_fit(cont_res, cont_err, sid, each_mjd, each_line + "-cont")
            flux_corr = corr_cont(wave_fit, flux_fit, fluxerr_fit, cont_res)
            if each_line == "Hbeta":
                try:
                    [fit_res, fit_err, figure_line] = hbeta_complex_fit(
                        wave_fit, flux_corr, fluxerr_fit)
                except SpectraException as reason:
                    print(str(reason))
                    exception_logging(sid, each_mjd, each_line, reason)
                    continue
            else:
                try:
                    [fit_res, fit_err, figure_line] = single_line_fit(
                        wave_fit, flux_corr, fluxerr_fit, line_set[each_line])
                except SpectraException as reason:
                    print(str(reason))
                    exception_logging(sid, each_mjd, each_line, reason)
                    continue
            figure_line.savefig("line-fig/"+str(sid)+"/"+str(each_mjd)+"-"+str(each_line)+".jpg")
            plt.close()
            try:
                rkk = check_fit(
                    wave_fit,
                    flux_corr,
                    fluxerr_fit,
                    fit_res,
                    each_line)
            except SpectraException as reason:
                print(str(reason))
                exception_logging(sid, each_mjd, each_line, reason)
                continue
            os.chdir("line/" + str(sid))
            try:
                os.mkdir(each_line)
            except OSError:
                pass
            os.chdir("../../")
            output_fit(fit_res, fit_err, sid, each_mjd, each_line)
            print("Process finished for " + each_line + "\n")
        print("Process finished for " + each_mjd + "\n")


line_set = {"C4": 1549.0, "Mg2": 2798.0, "Hbeta": 4902.0}
cont_set = {
    "C4": [
        1489.0, 1609.0], "Mg2": [
            2755.0, 2848.0], "Hbeta": [
                4750.0, 5100.0]}
#line_set = {"Hbeta": 4902.0}
#cont_set = {"Hbeta": [4750.0, 5100.0]}
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
        main_process(str(each_sid), line_set, cont_set)
    except Exception as reason:
        pass
