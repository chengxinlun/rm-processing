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
        error_fit.append(error[i])
    wave_fit = np.array(wave_fit)
    flux_fit = np.array(flux_fit)
    error_fit = np.array(error_fit)
    return [wave_fit, flux_fit, error_fit]


# Function to read in semi feII template and change them to completed fe2 template
def template_construct(fit_res):
    def fe2_template(x, ra = 1.0, deltax = 0.0, rstd = 1.0):
        template = models.Const1D(0)
        i = 0
        try:
            while True:
                template = template + models.Gaussian1D(ra * fit_res.parameters[i+1], deltax + fit_res.parameters[i+2], rstd * fit_res.parameters[i+3])
                i = i + 3
            except Exception:
                pass
        return template(x)
    template = models.custom_model(fe2_template)
    return template


# Function to fit Hbeta for se quasars
def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (Hbeta, FeII, OIII, OIII, FeII)
    hbeta_complex_fit_func = models.Lorentz1D(5.0, 4853.0, 40.0, bounds = {"amplitude": [0, 50.0], "x_0": [4833,4873]}) + \
            models.Gaussian1D(5.0, 4930.0, 1.0, bounds = {"amplitude": [0, 20.0], "mean": [4900, 4950]}) + \
            models.Gaussian1D(3.0, 4959.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4950, 4970]}) + \
            models.Gaussian1D(5.0, 4961.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4955, 4970]}) + \
            models.Gaussian1D(20.0, 5007.0, 6.0, bounds = {"amplitude": [0, 50.0], "mean": [4990, 5020]}) + \
            models.Gaussian1D(5.0, 5018.0, 7.0, bounds = {"amplitude": [0, 25.0], "mean": [5013, 5030]}) + \
            models.Linear1D((flux[0] - flux[-1])/(wave[0]-wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0])/(wave[0]-wave[-1]))
    #hbeta_complex_fit_func.mean_3.tied = lambda x: -48.0 + x.mean_4
    #hbeta_complex_fit_func.amplitude_3.tied = lambda x: 1.0 / 2.99 * x.amplitude_4
    #hbeta_complex_fit_func.stddev_3.tied = lambda x: 1.0 * x.stddev_4
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 10000)
        except Warning:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
            plt.plot(wave, cont(wave))
            fig.savefig("Hbeta-l-failed.jpg")
            plt.close()
            raise SpectraException("Line Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
    plt.plot(wave, cont(wave))
    fig.savefig("Hbeta-l.jpg")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters, rcs


# Function to fit Hbeta lines for non-se quasars
def hbeta_complex_fit_2(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    hbeta_complex_fit_func = \
            models.Gaussian1D(2.7, 4522.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4510, 4530]}) + \
            models.Gaussian1D(3.1, 4549.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4540, 4555]}) + \
            models.Gaussian1D(1.5, 4555.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4550, 4570]}) + \
            models.Gaussian1D(4.2, 4583.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4575, 4590]}) + \
            models.Gaussian1D(1.4, 4629.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4620, 4650]}) + \
            models.Gaussian1D(2.0, 4660.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4650, 4670]}) + \
            models.Gaussian1D(2.0, 4862.0, 1.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 4873], "stddev": [0.0, 7.0]}) + \
            models.Gaussian1D(5.0, 4860.0, 40.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 4873], "stddev": [7.0, 60.0]}) + \
            models.Gaussian1D(1.0, 4855.0, 70.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 5020]}) + \
            models.Gaussian1D(2.0, 4930.0, 1.0, bounds = {"amplitude": [0, 25.0], "mean": [4883, 4959]}) + \
            models.Gaussian1D(3.0, 4959.0, 12.0, bounds = {"amplitude": [0, 25.0], "mean": [4950, 4970]}) + \
            models.Gaussian1D(5.0, 4961.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4955, 4970]}) + \
            models.Gaussian1D(10.0, 5007.0, 3.0, bounds = {"amplitude": [0, 50.0], "mean": [4990, 5020]}) + \
            models.Gaussian1D(1.0, 5018.0, 7.0, bounds = {"amplitude": [0, 25.0], "mean": [5008, 5025]}) + \
            models.Gaussian1D(5.0, 5169.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5150, 5180]}) + \
            models.Gaussian1D(5.0, 5197.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5180, 5210]}) + \
            models.Gaussian1D(2.0, 5234.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5220, 5250]}) + \
            models.Gaussian1D(2.0, 5276.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [5260, 5300]}) + \
            models.Gaussian1D(5.0, 5316.0, 2.0, bounds = {"amplitude": [0, 10.0], "mean": [5300, 5325]}) + \
            models.PowerLaw1D(flux[0], wave[0], - np.log(flux[-1]/flux[0]) / np.log(wave[-1]/wave[0]), fixed = {"x_0": True})
    #hbeta_complex_fit_func.mean_5.tied = lambda x: -48.0 + x.mean_6
    #hbeta_complex_fit_func.amplitude_5.tied = lambda x: 1.0 / 2.99 * x.amplitude_6
    #hbeta_complex_fit_func.stddev_5.tied = lambda x: 1.0 * x.stddev_6
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 10000)
        except SpectraException:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.PowerLaw1D(fit.parameters[57], fit.parameters[58], fit.parameters[59])
            plt.plot(wave, cont(wave))
            plt.show()
            fig.savefig("Hbeta-g-failed.jpg")
            plt.close()
            raise SpectraException("Line Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.PowerLaw1D(fit.parameters[57], fit.parameters[58], fit.parameters[59])
    plt.plot(wave, cont(wave))
    plt.show()
    fig.savefig("Hbeta-g.jpg")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters, rcs


# Function to fit FeII lines before Hbeta
def fe2_before_hbeta(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (FeII, FeII, FeII), (FeII, FeII, FeII, FeII)
    hbeta_complex_fit_func = models.Gaussian1D(2.7, 4522.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4510, 4530]}) + \
            models.Gaussian1D(3.1, 4549.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4540, 4555]}) + \
            models.Gaussian1D(1.5, 4555.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4550, 4570]}) + \
            models.Gaussian1D(4.2, 4583.0, 5.0, bounds = {"amplitude": [0, 10.0], "mean": [4575, 4590]}) + \
            models.Gaussian1D(1.4, 4629.0, 7.0, bounds = {"amplitude": [0, 10.0], "mean": [4620, 4650]}) + \
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
            fig.savefig("bef-failed.jpg")
            plt.close()
            raise SpectraException("Line Fe2 before Hbeta fit failed")
    print(fit.parameters)
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
    plt.plot(wave, cont(wave))
    for i in range(5):
        fe = models.Gaussian1D(fit.parameters[3*i], fit.parameters[3*i+1], fit.parameters[3*i+2]) + cont
        plt.plot(wave, fe(wave))
    plt.show()
    fig.savefig("bef.jpg")
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
            fig.savefig("aft-failed.jpg")
            plt.close()
            raise SpectraException("Line Fe2 after Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[15], fit.parameters[16])
    plt.plot(wave, cont(wave))
    fig.savefig("aft.jpg")
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


# Add up flux
def flux_sum(part, wave, flux):
    sum_flux = 0
    for each in part:
        for i in range(len(wave)):
            if wave[i] < each[0]:
                continue
            if wave[i] > each[1]:
                break
            sum_flux = sum_flux + flux[i]
    return sum_flux


# Compare OIII and Fe2
def compare_fe2(wave, flux, error):
    # First fit Hbeta and OIII
    se = False
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 4400.0, 5400.0)
    try:
        [hbeta_g_res, g_rcs] = hbeta_complex_fit_2(wave_fit, flux_fit, error_fit)
    except SpectraException:
        se = True
        g_rcs = 65535
    try:
        [hbeta_l_res, l_rcs] = hbeta_complex_fit(wave_fit, flux_fit, error_fit)
    except SpectraException:
        if se==True:
            raise SpectraException("Fit for Hbeta complex failed")
        else:
            l_rcs = 65535
            pass
    if g_rcs>l_rcs:
        hbeta_res = hbeta_l_res
        o3range = [(hbeta_l_res[13] - 2.0 * hbeta_l_res[14], hbeta_l_res[13] + 2.0 * hbeta_l_res[14])]
        cont = lambda x: hbeta_l_res[18] * x + hbeta_l_res[19]
        hbetarange = [(hbeta_l_res[1] - 2.0 * hbeta_l_res[2], hbeta_l_res[1] + 2.0 * hbeta_l_res[2])]
    else:
        hbeta_res = hbeta_g_res
        o3range = [(hbeta_g_res[19] - 2.0 * hbeta_g_res[20], hbeta_g_res[19] + 2.0 * hbeta_g_res[20])]
        cont = lambda x: hbeta_g_res[24] * x +hbeta_g_res[25]
        hbetarange = union([(hbeta_g_res[1] - 2.0 * hbeta_g_res[2], hbeta_g_res[1] + 2.0 * hbeta_g_res[2]), 
            (hbeta_g_res[4] - 2.0 * hbeta_g_res[5], hbeta_g_res[4] + 2.0 * hbeta_g_res[5])])
    o3flux = flux_sum(o3range, wave, flux) - flux_sum(o3range, wave, list(map(cont, wave)))
    hbetaflux = flux_sum(hbetarange, wave, flux) - flux_sum(hbetarange, wave, list(map(cont, wave)))
    o3hb = o3flux / hbetaflux
    # Fit FeII lines before Hbeta complex
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 4400.0, 4720.0)
    bef_res = fe2_before_hbeta(wave_fit, flux_fit, error_fit)
    # Fit FeII lines after Hbeta complex
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 5120.0, 5400.0)
    aft_res = fe2_after_hbeta(wave_fit, flux_fit, error_fit)
    # Get total flux of each Fe line
    fecont = lambda x: bef_res[18] * x + bef_res[19]
    ferange = union([(bef_res[7] - 2.0 * bef_res[8], bef_res[7] + 2.0 * bef_res[8]),
        (bef_res[10] - 2.0 * bef_res[11], bef_res[10] + 2.0 * bef_res[11]),
        (bef_res[13] - 2.0 * bef_res[14], bef_res[13] + 2.0 * bef_res[14]),
        (bef_res[15] - 2.0 * bef_res[16], bef_res[15] + 2.0 * bef_res[16])])
    feflux = flux_sum(ferange, wave, flux) -  flux_sum(ferange, wave, list(map(cont, wave)))
    fecont = lambda x: aft_res[15] * x + aft_res[16]
    ferange = union([(aft_res[1] - 2.0 * aft_res[2], aft_res[1] + 2.0 * aft_res[2]),
        (aft_res[4] - 2.0 * aft_res[5], aft_res[4] + 2.0 * aft_res[5]),
        (aft_res[7] - 2.0 * aft_res[8], aft_res[7] + 2.0 * aft_res[8]),
        (aft_res[10] - 2.0 * aft_res[11], aft_res[10] + 2.0 * aft_res[11]),
        (aft_res[13] - 2.0 * aft_res[14], aft_res[13] + 2.0 * aft_res[14])])
    feflux = feflux + flux_sum(ferange, wave, flux) - flux_sum(ferange, wave, list(map(cont, wave)))
    fehb = feflux / hbetaflux
    return [hbeta_res, bef_res, aft_res, o3hb, fehb]


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
    print(o3sn, fesn)
    print("Process finished for " + str(sid))


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
#sid_list = [521, 1039]
sid_list = [1141]
for each_sid in sid_list:
    try:
        main_process(str(each_sid), line)
    except Exception as reason:
        print(str(reason))
        pass
