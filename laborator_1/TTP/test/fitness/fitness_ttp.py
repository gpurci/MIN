#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules
sys_remove_modules("extension.utils.my_code.normalization")

from extension.fitness.my_code.fitness_base import *
from extension.utils.my_code.normalization import *

class FitnessTTP(FitnessBase):
    """
    Clasa 'FitnessTTP', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="FitnessTTP", **configs)
        self.__fn = self._unpackMethod(method, 
                                        f1score=self.f1score,
                                        linear=self.linear,
                                        norm_linear=self.norm_linear,
                                        damped=self.damped,
                                        exponential=self.exponential,
                                        norm_log=self.norm_log
                                    )

    def __call__(self, metric_values):
        return self.__fn(metric_values, **self._configs)

    def help(self):
        info = """FitnessTTP:
    metoda: 'f1score';     config: -> P_pres=1, W_pres=1, T_pres=1, D_pres=1;
    metoda: 'linear';      config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'norm_linear'; config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'damped';      config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'exponential'; config: -> W_pres=1, D_pres=1, R=1;
    metoda: 'norm_log';    config: -> W_pres=1, D_pres=1, R=1;\n"""
        print(info)

    # TTP ------------------------------
    def f1score(self, metric_values, P_pres=1, W_pres=1, T_pres=1, D_pres=1):
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

    def linear(self, metric_values, W_pres=1, D_pres=1, R=1):
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
        weights   = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        fitness = mask_city * number_obj * weights * distances * (profits - R * times + 1e-7)
        return fitness

    def norm_linear(self, metric_values, W_pres=1, D_pres=1, R=1):
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

    def damped(self, metric_values, W_pres=1, D_pres=1, R=1):
        """
        """
        # unpack metrics
        profits = metric_values.get("profits")
        weights = metric_values.get("weights") # normalized
        times   = metric_values.get("times")
        distances   = metric_values.get("distances", np.array([1.], dtype=np.float32))
        number_city = metric_values.get("number_city")
        number_obj  = metric_values.get("number_obj")
        # presure
        distances   = min_nonzeronorm(distances)
        distances   = distances**D_pres
        weights = weights**W_pres
        mask_city = self.__cityBinarise(number_city)
        number_obj= number_obj / self.GENOME_LENGTH
        # calculate fitness
        fitness = mask_city * number_obj * weights * distances * profits * np.exp(-R * times)
        return fitness

    def exponential(self, metric_values, W_pres=1, D_pres=1, R=1):
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
        fitness = mask_city * number_obj * weights * distances * np.exp(profits - R * times)
        return fitness

    def norm_log(self, metric_values, W_pres=1, D_pres=1, R=1):
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
        fitness = mask_city * number_obj * weights * distances * np.exp(profits) / (1 + np.exp(R * times))
        return fitness

    # TTP =================================

    def __cityBinarise(self, number_city):
        mask_cities = (number_city>=self.GENOME_LENGTH).astype(np.float32)
        return mask_cities
