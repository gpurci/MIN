#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules
sys_remove_modules("extension.utils.normalization")
sys_remove_modules("extension.utils.standardization")

from extension.fitness.fitness_base import *
from extension.utils.normalization import *
from extension.utils.standardization import *

class FitnessTTPV2(FitnessBase):
    """
    Clasa 'FitnessTTPV2', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="FitnessTTPV2", **configs)
        self.__fn = self._unpackMethod(method, 
                                        f1score=self.f1score,
                                        linear=self.linear,
                                        norm_linear=self.norm_linear,
                                        mixt=self.mixt,
                                    )
        self.reference = {
            "profits"    : [0, 60000],
            "times"      : [2500, 2500],
            "distances"  : [2500, 2500],
            "score"      : [-20000, 20000]
            }

    def __call__(self, metric_values):
        profits = metric_values.get("profits")
        weights = metric_values.get("weights") # normalized
        times   = metric_values.get("times")
        distances   = metric_values.get("distances")
        self.reference["profits"][0] = self.reference["profits"][0]*0.9 + 0.1*profits.min()
        self.reference["profits"][1] = self.reference["profits"][1]*0.9 + 0.1*profits.max()

        self.reference["times"][0] = self.reference["times"][0]*0.9 + 0.1*min_nonzero(times)
        self.reference["times"][1] = self.reference["times"][1]*0.9 + 0.1*min_nonzero(times)

        self.reference["distances"][0] = self.reference["distances"][0]*0.9 + 0.1*min_nonzero(distances)
        self.reference["distances"][1] = self.reference["distances"][1]*0.9 + 0.1*min_nonzero(distances)
        
        return self.__fn(metric_values, **self._configs)

    def help(self):
        info = """FitnessTTPV2:
    metoda: 'f1score';     config: -> P_pres=1, W_pres=1, T_pres=1, D_pres=1;
    metoda: 'linear';      config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'norm_linear'; config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'mixt';        config: -> p_select=[1/3, 1/3, 1/3], P_pres=1, W_pres=1, T_pres=1, D_pres=1, R=1;\n"""
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
        profits = normalization_reference(profits, self.reference["profits"])
        profits = profits**P_pres
        weights = weights**W_pres
        times   = min_nonzeronorm_reference(times, self.reference["times"])
        times   = times**T_pres
        distances   = min_nonzeronorm_reference(distances, self.reference["distances"])
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
        tmp_score = profits - R * times
        min_score = tmp_score.min()
        tmp_score -= min_score
        fitness = mask_city * tmp_score
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
        distances = min_nonzeronorm_reference(distances, self.reference["distances"])
        distances = distances**D_pres
        weights   = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        tmp_score = profits - R * times
        tmp_score = standardization_reference(tmp_score, self.reference["score"])
        fitness = mask_city * number_obj * weights * distances * tmp_score / (weights + distances)
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


    # TTP =================================

    def __cityBinarise(self, number_city):
        mask_cities = (number_city>=self.GENOME_LENGTH).astype(np.float32)
        return mask_cities
