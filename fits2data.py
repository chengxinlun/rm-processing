# Module import
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from psrm.base.readspec import readspec
from psrm.base.target_fibermap import parseVar_sid, parseVar_pmf
from psrm.analSpec.ob2rf import ob2rf


# Extract information from fits to a database made from .pkl
def info_extraction():
    info = parseVar_sid('RM', 'ra', 'dec', 'zfinal', 'plate', 'mjd', 'fiberid')
    try:
        os.mkdir("info_database")
    except OSError:
        pass
    # Using pickle file to store the dictionary
    store_database(info, 'ra')
    store_database(info, 'dec')
    store_database(info, 'zfinal')
    sid_list = info.keys()
    os.chdir("info_database")
    sidfile = open("sid.pkl", "wb")
    pickle.dump(sid_list, sidfile)
    sidfile.close()
    os.chdir("../")
    for each_sid in sid_list:
        mjd = info[each_sid]['mjd']
        plate = info[each_sid]['plate']
        fiberid = info[each_sid]['fiberid']
        pmf_linker(each_sid, mjd, plate, fiberid)


def store_database(mother_dict, item):
    sid_list = mother_dict.keys()
    itemfile = open("info_database/" + item + ".pkl", "wb")
    itemdict = dict()
    for each_sid in sid_list:
        itemdict[each_sid] = mother_dict[each_sid][item]
    pickle.dump(itemdict, itemfile)
    itemfile.close()
    print("Database for " + item + " constructed.\n")


# Link mjd with plate and fiberid
def pmf_linker(sid, mjd, plate, fiberid):
    try:
        os.mkdir("pmf_database")
    except OSError:
        pass
    os.chdir("pmf_database")
    datafile = open(str(sid) + ".pkl", "wb")
    datalist = list()
    for i in range(len(mjd)):
        datalist.append(str(mjd[i]).strip("\n") +
                        "-" +
                        str(plate[i]).strip("\n") +
                        "-" +
                        str(fiberid[i]).strip("\n"))
    pickle.dump(datalist, datafile)
    datafile.close()
    print("Link complete for " + str(sid) + "\n")
    os.chdir("../")


# Recommended access of pmf_database
def read_pmf_database(sid):
    readfile = open("pmf_database/" + str(sid) + ".pkl", "rb")
    pmf = pickle.load(readfile)
    plate = list()
    fiberid = list()
    mjd = list()
    for each in pmf:
        mjd.append(each.split("-")[0])
        plate.append(each.split("-")[1])
        fiberid.append(each.split("-")[2])
    return [mjd, plate, fiberid]


# Recommended access of sid in info_database
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


# Recommended output method of any data in the form of general list
def output(data_list, out_file):
    pickle_file = open(out_file, 'wb')
    pickle.dump(data_list, pickle_file)
    pickle_file.close()


# Read and record data
def read_fits(plate, fiberid, mjd, sid):
    try:
        os.mkdir("raw")
    except OSError:
        pass
    os.chdir("raw")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir(str(sid))
    try:
        os.mkdir(str(mjd))
    except OSError:
        pass
    os.chdir(str(mjd))
    # Read data from fits (code derived from gaoyang's plotSpec.py)
    parM = readspec(
        plate,
        fiberid,
        'flux',
        'loglam',
        'wave',
        'fluxerr',
        mjd=mjd)
    zs = parseVar_pmf(plate, mjd, fiberid, 'zfinal', 'sourcetype')
    key = zs.keys()[0]
    zfinal = zs[key]['zfinal']
    sourcetype = zs[key]['sourcetype']
    wave = parM['wave']
    flux = parM['flux']
    fluxerr = parM['fluxerr']
    # Restframe correction
    rf = ob2rf(wave, flux, zfinal, fluxerr=fluxerr)
    # Output to file
    output(rf["wave"], "wave.pkl")
    output(rf["flux"], "flux.pkl")
    output(rf["fluxerr"], "fluxerr.pkl")
    os.chdir("../../../")
    print("Process complete for " + str(sid) + " " + str(mjd))


# Main process


def main_process():
    info_extraction()
    sid_list = get_total_sid_list()
    for each in sid_list:
        [mjd, plate, fiberid] = read_pmf_database(each)
        print(fiberid)
        for i in range(len(mjd)):
            read_fits(int(plate[i]), int(fiberid[i]), int(mjd[i]), each)


main_process()
