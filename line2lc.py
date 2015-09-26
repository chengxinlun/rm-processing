import os
import random
import matplotlib.pylab as plt
import numpy as np
import thread
import pickle


class SpectraException(Exception):
    pass


# Recommended function to read the fit result
def read_fit_result(sid, line, mjd):
    try:
        linefile = open("line/" + str(sid) + "/" + str(line) + "/" + str(mjd) + ".pkl", 'rb')
    except IOError:
        raise SpectraException("Unable to find line file")
    try:
        contfile = open(
            "line/" +
            str(sid) +
            "/" + str(line) +
            "-cont/" +
            str(mjd) +
            ".pkl",
            'rb')
    except IOError:
        raise SpectraException("Unable to find cont file")
    line_fit_res = pickle.load(linefile)
    line_fit_err = pickle.load(linefile)
    cont_fit_res = pickle.load(contfile)
    cont_fit_err = pickle.load(contfile)
    return [line_fit_res, line_fit_err, cont_fit_res, cont_fit_err]


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


# Continuum correction
def calc_continuum(line, fit_result):
    cont_func = lambda x: fit_result[0] * (x**fit_result[1])
    return cont_func(line)


# Exception logging process
def exception_logging(sid, mjd, line, reason):
    log = open("add_error.log", "a")
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


# Main process
def main_process():
    line_set = {"C4": 1600.0, "N5": 1300.0, "Mg2": 2900.0, "Hbeta": 5100.0}
    sid_list = get_total_sid_list()
    for each_sid in sid_list:
        for each_line in line_set.keys():
            try:
                mjd_list = os.listdir(
                    "line/" +
                    str(each_sid) +
                    "/" +
                    each_line)
            except Exception:
                print("Cannot find " + each_line + " for " + str(each_sid))
                continue
            for mjd in mjd_list:
                each_mjd = mjd.strip(".pkl")
                [wave, flux, fluxerr] = read_raw_data(each_sid, each_mjd)
                try:
                    [line_fit_res,
                     line_fit_err,
                     cont_fit_res,
                     cont_fit_err] = read_fit_result(each_sid,
                                                     each_line,
                                                     each_mjd)
                except SpectraException as reason:
                    print(str(reason))
                    exception_logging(each_sid, each_mjd, each_line, reason)
                    continue
                [wave, flux, fluxerr] = mask_points(wave, flux, fluxerr)
                cont_res = calc_continuum(line_set[each_line], cont_fit_res)
                if each_line != "Hbeta":
                    [wave_calc,
                     flux_calc,
                     fluxerr_calc] = extract_fit_part(wave,
                                                      flux,
                                                      fluxerr,
                                                      line_fit_res[1],
                                                      2.0 * abs(line_fit_res[2]))
                else:
                    min_wave = line_fit_res[4] - 2.0 * abs(line_fit_res[5])
                    max_wave = min([line_fit_res[1] -
                                    2.0 *
                                    abs(line_fit_res[2]), line_fit_res[4] +
                                    2.0 *
                                    abs(line_fit_res[5]), line_fit_res[7] -
                                    2.0 *
                                    abs(line_fit_res[2])])
                    [wave_calc,
                     flux_calc,
                     fluxerr_calc] = extract_fit_part(wave,
                                                      flux,
                                                      fluxerr,
                                                      0.5 * (min_wave + max_wave),
                                                      0.5 * (max_wave - min_wave))
                flux_calc_corr = list()
                for i in range(len(wave_calc)):
                    flux_calc_corr.append(
                        flux_calc[i] -
                        calc_continuum(
                            wave_calc[i],
                            cont_fit_res))
                line_res = sum(flux_calc_corr)
                line_err = sum(fluxerr_calc)
                cont_err = np.mean(fluxerr_calc)
                output_linefile = open(
                    "lc/" +
                    str(each_sid) +
                    "/" +
                    str(each_line) +
                    ".txt",
                    "a")
                output_linefile.write(
                    "%d    %9.4f    %9.4f\n" %
                    (int(each_mjd), float(line_res), float(line_err)))
                output_linefile.close()
                output_contfile = open(
                    "lc/" +
                    str(each_sid) +
                    "/" +
                    str(each_line) +
                    "-cont.txt",
                    "a")
                output_contfile.write(
                    "%d    %9.4f    %9.4f\n" %
                    (int(each_mjd), float(cont_res), float(cont_err)))
                output_contfile.close()

try:
    os.mkdir("lc")
except OSError:
    pass
target_list = get_total_sid_list()
os.chdir("lc")
for each in target_list:
    try:
        os.mkdir(str(each))
    except OSError:
        pass
os.chdir("../")
main_process()
