import os
import pickle
import numpy as np
from astropy.modeling import models, fitting, Fittable1DModel, Parameter


class FeII_template(Fittable1DModel):


    inputs = ('x',)
    outputs = ('y',)

    temperature = Parameter()
    shift = Parameter()
    width = Parameter()
    i_F = Parameter()
    i_S = Parameter()
    i_G = Parameter()
    i_P = Parameter()
    i_lzw1 = Parameter()
    

    @staticmethod
    def evaluate(x, temperature, shift, width, i_F, i_S, i_G, i_P, i_lzw1):
	def reading_fe2_info(group):
            filein = open("fe2_template_info/"+group+".txt")
            center = list()
            gf = list()
            deltae = list()
            for each_line in filein:
                center.append(float(each_line.split("  ")[0]))
                gf.append(float(each_line.split("  ")[1]))
                deltae.append(float(each_line.split("  ")[2]))
            filein.close()
            return [center, gf, deltae]
    

    	def calc_i_ratio(center, gf, deltae, temp, index):
            i_ratio = list(range(len(center)))
            for i in range(len(center)):
                i_ratio[i] = gf[i] / (center[i] ** 3) * np.exp(1.43867173377 * deltae[i] / temp)
            standard = i_ratio[index]
            for i in range(len(center)):
                i_ratio[i] = i_ratio[i] / standard
            return i_ratio
    

        [center_F, gf_F, deltae_F] = reading_fe2_info("F")
        [center_S, gf_S, deltae_S] = reading_fe2_info("S")
        [center_G, gf_G, deltae_G] = reading_fe2_info("G")
        [center_P, gf_P, deltae_P] = reading_fe2_info("P")
        center_lzw1 = [4418.95703125, 4449.61621094, 4471.27294922, 4493.52880859, 4614.55078125, 4625.48095703, 4628.78613281, 4631.87304688,
            4660.59277344, 4668.92285156, 4740.82812500, 5131.20996094, 5369.18994141, 5396.23193359, 5427.82617188]
        i_r_lzw1 = [2.27002811, 1.13501406, 0.90801126, 1.21068168, 0.52967322, 0.52967322, 0.90801126, 0.45400563, 0.75667602, 0.68100840,
            0.37833801, 0.83234364, 1.09718025, 0.30267042, 1.05934644]


        i_r_F = calc_i_ratio(center_F, gf_F, deltae_F, temperature, 9)
        i_r_S = calc_i_ratio(center_S, gf_S, deltae_S, temperature, 3)
        i_r_G = calc_i_ratio(center_G, gf_G, deltae_G, temperature, 4)
        i_r_P = calc_i_ratio(center_P, gf_P, deltae_P, temperature, 4)
        res = models.Gaussian1D(i_lzw1 * i_r_lzw1[0], center_lzw1[0], width * np.sqrt(3/2)*center_lzw1[0]/299792.458)
        for i in range(1,len(center_lzw1)):
            res = res + models.Gaussian1D(i_lzw1 * i_r_lzw1[i], center_lzw1[i], width*np.sqrt(3/2)*center_lzw1[i]/299792.458)
        for i in range(len(center_F)):
            res = res + models.Gaussian1D(i_F * i_r_F[i], center_F[i], width*np.sqrt(3/2)*center_F[i]/299792.458)
        for i in range(len(center_S)):
            res = res + models.Gaussian1D(i_S * i_r_S[i], center_S[i], width*np.sqrt(3/2)*center_S[i]/299792.458)
        for i in range(len(center_G)):
            res = res + models.Gaussian1D(i_G * i_r_G[i], center_G[i], width*np.sqrt(3/2)*center_G[i]/299792.458)
        for i in range(len(center_P)):
            res = res + models.Gaussian1D(i_P * i_r_P[i], center_P[i], width*np.sqrt(3/2)*center_P[i]/299792.458)
        return res(x)

if __name__ != "main":
    pass
