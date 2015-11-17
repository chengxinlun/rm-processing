import os
import numpy as np
import pickle
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

def read_in_curves(width, group):
    filein = open("FeII_template_4000_5500/"+width+"FeII/fe_"+group+".txt")
    wave =  []
    flux = []
    for each in filein:
        wave.append(float(each.split("     ")[0]))
        flux.append(float(each.split("     ")[1]))
    filein.close()
    return [wave, flux]


def read_in_group(group):
    filein = open("FeII_template_4000_5500/FeII_rel_int/"+group+"_rel_int.txt")
    fe2models = models.Const1D(0, fixed = {"amplitude": True})
    for each in filein:
        fe2models = fe2models + models.Gaussian1D(float(each.split("     ")[1]), float(each.split("     ")[0]), 4.0 , fixed = {"mean": True})
    filein.close()
    return fe2models


def fit_standard_curve(wave, flux, standard):
    fig = plt.figure()
    plt.plot(wave, flux)
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(standard, wave, flux, maxiter = 100000)
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    return [fit, fig]


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
    template._is_dynamic = True
    return template


def output_template(width, group, template, fig):
    os.chdir("FeII_template")
    try:
        os.mkdir(width)
    except OSError:
        pass
    os.chdir(width)
    fileout = open(group+".pkl", "wb")
    pickle.dump(template, fileout)
    fileout.close()
    fig.savefig(group+".jpg")
    plt.close()
    os.chdir("../../")


def main_process(width, group):
    [wave,flux] = read_in_curves(width, group)
    standard = read_in_group(group)
    [res, fig] = fit_standard_curve(wave, flux, standard)
    print(res.parameters)
    output_template(width, group, res, fig)
    print("Template constructed for " + width + " " + group)


width_list = ['700', '800', '900' ,'1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800']
group_list = ['f', 'g', 'IZw1', 'p', 's']
for width in width_list:
    for group in group_list:
        main_process(width, group)
