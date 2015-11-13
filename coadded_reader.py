import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits


# Reader specified for Sun Jiayi's coadded spectra
def read_object(fits_file):
    filein = fits.open(fits_file)
    overall_data = filein[1].data
    ra = overall_data.field('RA')[0]
    dec = overall_data.field('DEC')[0]
    indx = overall_data.field('INDX')[0]
    zfinal = overall_data.field('ZFINAL')[0]
    wave = overall_data.field('REST_WAVE')[0]
    flux = overall_data.field('DERED_FLUX')[0]
    inverse_var = overall_data.field('DERED_IVAR')[0]
    filein.close()
    return [ra, dec, indx, zfinal, wave, flux, inverse_var]


def output(data_list, out_file):
    pickle_file = open(out_file, 'wb')
    pickle.dump(data_list, pickle_file)
    pickle_file.close()


def process_object(fits_file):
    [ra, dec, indx, zfinal, wave, flux, inverse_var] = read_object(fits_file)
    os.chdir("raw")
    try:
        os.mkdir(str(indx))
    except OSError:
        pass
    os.chdir(str(indx))
    output(wave, "wave.pkl")
    output(flux, "flux.pkl")
    output(inverse_var, "ivar.pkl")
    os.chdir("../../")
    return [ra, dec, zfinal, indx]


def main_process():
    try:
        os.mkdir("info_database")
    except:
        pass
    try:
        os.mkdir("raw")
    except:
        pass
    coadded_dir = "/home/cheng/coadded"
    fits_list = os.listdir(coadded_dir)
    sid_list = list()
    ra_dict = dict()
    dec_dict = dict()
    zfinal_dict = dict()
    for each_fits in fits_list:
        [ra, dec, zfinal, indx] = process_object(coadded_dir + "/" + each_fits)
        ra_dict[indx] = ra
        dec_dict[indx] = dec
        zfinal_dict[indx] = zfinal
        sid_list.append(indx)
    output(sid_list, "info_database/sid.pkl")
    output(ra_dict, "info_database/ra.pkl")
    output(dec_dict, "info_database/dec.pkl")
    output(zfinal_dict, "info_database/zfinal.pkl")

main_process()
