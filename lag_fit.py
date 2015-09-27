import numpy as np
from scipy.optimize import curve_fit
import os
import pickle
import matplotlib.pyplot as plt


# Make the histogram of the file returned by Javelin
def count_file(file_in):
    bef_hist = list()
    infile = open(file_in)
    for each_line in infile:
        bef_hist.append(float(each_line.split(" ")[2]))
    # For debug purpose only
    #plt.hist(bef_hist, 70, range=(0.0, max(bef_hist)))
    return np.histogram(bef_hist, bins=70, range=(0.0, max(bef_hist)))


# Fitting the histogram with a gaussian function
def gauss_fit_hist(hist, bin_med):
    max_loc = np.argmax(hist)
    gauss = lambda x, a, x0, sig: a * np.exp(-(x - x0)**2 / (2 * sig**2))
    if max_loc > 3:
        hist_fit = hist[(max_loc - 3):(max_loc + 3)]
        bin_med_fit = bin_med[(max_loc - 3):(max_loc + 3)]
    else:
        hist_fit = hist[0:5]
        bin_med_fit = bin_med[0:5]
    guess = [hist[max_loc], bin_med[max_loc], np.std(hist_fit)]
    [fit_res, fit_extra] = curve_fit(
        gauss, bin_med_fit, hist_fit, p0=guess, maxfev=10000)
    fit_err = np.sqrt(np.diag(fit_extra))
    if fit_err[1] > 20.0:
        fit_res[1] = bin_med[max_loc]
        fit_err[1] = 0.5000
    if fit_res[1] <= 0.1:
        raise Exception
    # For debug purpose only
    #fit_func = lambda x: gauss(x, fit_res[0], fit_res[1], fit_res[2])
    #plt.plot(bin_med_fit, list(map(fit_func, bin_med_fit)))
    #plt.show()
    return [fit_res, fit_err]


# Calculate the continuum from lc\sid
def calc_cont(file_in):
    infile = open(file_in)
    flux = list()
    error = list()
    for each_line in infile:
        flux.append(float(each_line.split("    ")[1]))
        error.append(float(each_line.split("    ")[2]))
    if np.mean(flux) <= 0.01:
        raise Exception
    return [np.mean(flux), np.mean(error)]


# Get the whole sid list
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


def main_process(line_set, sid):
    for each_line in line_set:
        print("Processing " + str(sid))
        os.chdir("lc/" + str(sid))
        try:
            [hist, bin_med] = count_file("cont-" + each_line + ".txt")
        except IOError:
            print("Unable to find " + str(sid) + " " + each_line)
            os.chdir("../../")
            continue
        try:
            [fit_res, fit_err] = gauss_fit_hist(hist, bin_med)
        except Exception:
            os.chdir("../../")
            print("Unable to find " + str(sid) + " " + each_line)
            continue
        try:
            [cont_res, cont_err] = calc_cont(each_line + "-cont.txt")
        except Exception:
            os.chdir("../../")
            print("Unable to find " + str(sid) + " " + each_line + "cont")
            continue
        os.chdir("../../lag")
        fitfile = open(each_line + ".txt", "a")
        fitfile.write("%9.4f    %9.4f    %d\n" % (fit_res[1], fit_err[1], sid))
        fitfile.close()
        outfile_name = each_line + "-cont.txt"
        outfile = open(outfile_name, "a")
        outfile.write("%9.4f    %9.4f    %d\n" % (cont_res, cont_err, sid))
        outfile.close()
        os.chdir("../")


try:
    os.mkdir("lag")
except OSError:
    pass
line_set = ["Hbeta", "C4", "N5", "Mg2"]
sid_list = get_total_sid_list()
for each_sid in sid_list:
    main_process(line_set, each_sid)

