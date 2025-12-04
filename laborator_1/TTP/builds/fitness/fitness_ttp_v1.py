#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules
sys_remove_modules("extension.utils.normalization")

from extension.fitness.fitness_base import *
from extension.utils.normalization import *

class FitnessTTPV1(FitnessBase):
    """
    Clasa 'FitnessTTPV1', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, cycles=None, **configs):
        super().__init__(method, name="FitnessTTPV1", **configs)
        self.__fn = self._unpackMethod(method, 
                                        f1score=self.f1score,
                                        linear=self.linear,
                                        norm_linear=self.norm_linear,
                                        mixt=self.mixt,
                                        cyclic=self.cyclic
                                    )
        self.__cycles = self.__check_cyclic(cycles)
        self.__count  = 0

    def __check_cyclic(self, cyclic):
        ret = []
        if (cyclic is None):
            ret = [1, 2, 3]
        else:
            if (len(cyclic) != 3):
                raise NameError("Lungimea la 'cyclic' '{}', este mai mare decat '3'!".format(cyclic))
            tmp = 0
            for c in cyclic:
                ret.append(c+tmp)
                tmp += c
        return ret

    def __call__(self, metric_values):
        return self.__fn(metric_values, **self._configs)

    def help(self):
        info = """FitnessTTPV1:
    metoda: 'f1score';     config: -> P_pres=1, W_pres=1, T_pres=1, D_pres=1;
    metoda: 'linear';      config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'norm_linear'; config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'mixt';        config: -> p_select=[1/3, 1/3, 1/3], P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1;
    metoda: 'cyclic';      config: -> cycles=[1, 1, 1], P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1;\n"""
        print(info)

    # TTP ------------------------------
    def f1score(self, metric_values, P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1):
        """
        """
        # unpack metrics
        profits = metric_values.get("profits")
        weights = metric_values.get("weights") # normalized
        times   = metric_values.get("times")
        distances   = metric_values.get("distances", np.array([1.], dtype=np.float32))
        number_city = metric_values.get("number_city")
        number_obj  = metric_values.get("number_obj")
        # normalization
        profits = normalization(profits)
        profits = profits**P_pres
        weights = weights**W_pres
        times   = min_nonzeronorm(times)
        times   = times**T_pres
        distances   = min_nonzeronorm(distances)
        distances   = distances**D_pres
        mask_city = self.__cityBinarise(number_city)
        number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        fitness = mask_city * number_obj * ((weights * profits * distances * times) / (weights + profits + distances + times + 1e-7))
        return fitness

    def linear(self, metric_values, P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1):
        """
        """
        # unpack metrics
        profits = metric_values.get("profits")
        #weights = metric_values.get("weights") # normalized
        times   = metric_values.get("times")
        #distances   = metric_values.get("distances", np.array([1.], dtype=np.float32))
        number_city = metric_values.get("number_city")
        #number_obj  = metric_values.get("number_obj")
        # normalization
        #distances   = min_nonzeronorm(distances)
        #   = distances**D_pres
        #weights   = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        #number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        fitness = mask_city * (profits - R * times + 1e-7)
        return fitness

    def norm_linear(self, metric_values, P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1):
        """
        """
        # unpack metrics
        profits = metric_values.get("profits")
        weights = metric_values.get("weights") # normalized
        times   = metric_values.get("times")
        distances   = metric_values.get("distances", np.array([1.], dtype=np.float32))
        number_city = metric_values.get("number_city")
        number_obj  = metric_values.get("number_obj")
        # normalization
        distances   = min_nonzeronorm(distances)
        distances   = distances**D_pres
        weights = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        tmp_linear = (profits - R * times + 1e-7)
        linear_min = tmp_linear.min()
        linear_max = tmp_linear.max()
        norm_linear = (tmp_linear - linear_min) / (linear_max - linear_min + 1e-7)
        fitness = mask_city * number_obj * weights * distances * norm_linear / (weights + distances + norm_linear)
        return fitness

    def mixt(self, metric_values, p_select=None, **kw):
        """
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.f1score(    metric_values, **kw)
        elif (cond == 1):
            offspring = self.linear(     metric_values, **kw)
        elif (cond == 2):
            offspring = self.norm_linear(metric_values, **kw)
        return offspring

    def cyclic(self, metric_values, **kw):
        """
        """
        if   (self.__count <= self.__cycles[0]):
            offspring = self.f1score(    metric_values, **kw)
        elif (self.__count <= self.__cycles[1]):
            offspring = self.linear(     metric_values, **kw)
        elif (self.__count <= self.__cycles[2]):
            offspring = self.norm_linear(metric_values, **kw)
        else:
            self.__count = 1
            offspring = self.f1score(    metric_values, **kw)
        self.__count +=1
        return offspring


    # TTP =================================

    def __cityBinarise(self, number_city):
        mask_cities = (number_city>=self.GENOME_LENGTH).astype(np.float32)
        return mask_cities
