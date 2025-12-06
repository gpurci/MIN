#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules
sys_remove_modules("extension.utils.normalization")
sys_remove_modules("extension.utils.standardization")

from extension.fitness.fitness_base import *
from extension.utils.normalization import *
from extension.utils.standardization import *

class FitnessTTPV3(FitnessBase):
    """
    Clasa 'FitnessTTPV3', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="FitnessTTPV3", **configs)
        self.__fn = self._unpackMethod(method, 
                                        f1score=self.f1score,
                                        linear=self.linear,
                                        norm_linear=self.norm_linear,
                                        mixt=self.mixt,
                                    )
        self.prev_metric_values = None

    def __call__(self, metric_values):
        return self.__fn(metric_values, **self._configs)

    def help(self):
        info = """FitnessTTPV3:
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
        score = profits - R * times
        if (self.prev_metric_values is not None):
            x_min, x_max = normal_values(metric_values, self.prev_metric_values, "profits")
            profits   = normalization_reference(profits, x_min, x_max)
            x_min = min_nonzero_values(metric_values, self.prev_metric_values, "times")
            times     = min_nonzeronorm_reference(times, x_min)
            x_min = min_nonzero_values(metric_values, self.prev_metric_values, "distances")
            distances = min_nonzeronorm_reference(distances, x_min)
        else:
            profits   = normalization(profits)
            times     = min_nonzeronorm(times)
            distances = min_nonzeronorm(distances)
        # normalization
        profits   = profits**P_pres
        weights   = weights**W_pres
        times     = times**T_pres
        distances = distances**D_pres
        mask_city = self.__cityBinarise(number_city)
        # calculate fitness
        fitness = mask_city * ((weights * profits * distances * times) / (weights + profits + distances + times + 1e-7))
        x_min, x_max = norm_score(score, self.prev_metric_values)
        self.prev_metric_values = metric_values
        self.prev_metric_values["scores"] = (x_min, x_max)
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
        score = profits - R * times
        fitness = mask_city * score
        x_min, x_max = norm_score(score, self.prev_metric_values)
        self.prev_metric_values = metric_values
        self.prev_metric_values["scores"] = (x_min, x_max)
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
        if (self.prev_metric_values is not None):
            x_min = min_nonzero_values(metric_values, self.prev_metric_values, "distances")
            distances = min_nonzeronorm_reference(distances, x_min)
        else:
            distances = min_nonzeronorm(distances)
        # normalization
        distances = distances**D_pres
        weights   = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        # calculate fitness
        score = profits - R * times
        x_min, x_max = norm_score(score, self.prev_metric_values)
        score = normalization_reference(score, x_min, x_max)
        fitness = mask_city * weights * distances * score / (weights + distances + score)
        self.prev_metric_values = metric_values
        self.prev_metric_values["scores"] = (x_min, x_max)
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

def norm_score(score, metric_values):
    if (metric_values is not None):
        x_min, x_max = metric_values.get("scores", (None, None))
    else:
        x_min = None
    if (x_min is not None):
        x_min, x_max = min(x_min, score.min()), max(x_max, score.max())
    else:
        x_min, x_max = score.min(), score.max()
    return x_min, x_max
