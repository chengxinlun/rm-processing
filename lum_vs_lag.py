import numpy as np
from scipy.stats import linregress
import os
import pickle
import matplotlib.pyplot as plt


#Read all data from pickle file
def read_all(file_in):
    infile = open(file_in, "rb")
    lum = pickle.load(infile)
    lum_err = pickle.load(infile)
    lag = pickle.load(infile)
    lag_err = pickle.load(infile)
    infile.close()
    lum_err = np.array(lum_err) / (np.array(lum) * np.log(10.0))
    lag_err = np.array(lag_err) / (np.array(lag) * np.log(10.0))
    lum = np.log10(lum)
    lag = np.log10(lag)
    return [lum, lum_err, lag, lag_err]


# Main process
def main_process(line):
    [lum, lum_err,lag, lag_err]=read_all("corrected/"+line+".pkl")
    [k, b, r, p, stderr]=linregress(lum, lag)
    fit_fn=lambda x: k*x+b
    x = [min(lum), max(lum)]
    plt.plot(x, list(map(fit_fn,x)))
    print(line)
    print(k,b,r*r)
    plt. errorbar(lum, lag, xerr = lum_err, yerr = lag_err, linewidth = 0.0, elinewidth = 0.5)
    plt.show()


line_set = ["Hbeta", "C4", "Mg2", "N5"]
for each in line_set:
    main_process(each)
