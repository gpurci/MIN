#!/usr/bin/python

import numpy as np
from extension.fitness.my_code.fitness_base import *

class FitnessTTP(FitnessBase):
    """
    Clasa 'FitnessTTP', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="FitnessTTP", **configs)
        self.__fn = self._unpackMethod(method, 
                                        TTP_f1score=self.fitnessF1scoreTTP, 
                                        TTP=self.fitnessTTP,
                                    )

    def __call__(self, metric_values):
        return self.__fn(metric_values, **self._configs)

    def help(self):
        info = """Fitness:
    metoda: 'TTP_f1score'; config: -> "R":1, ;
    metoda: 'TTP';         config: -> "R":1, ;\n"""
        print(info)

    # TTP ------------------------------
    def fitnessTTP(self, metric_values, R=1):
        """
        Fitness cu decadere liniara.
        Pentru fiecare individ:
        - mergem pe traseu (route)
        - cand ajungem intr-un oras luam obiectele de acolo
        - profitul scade liniar cu timpul:
                p(t) = p0 - alpha * t
        - viteza scade in functie de greutatea acumulata
        - costul de timp este penalizat cu R
        Returneaza:
            vector np.array cu fitness pentru fiecare individ
        """
        # unpack metrics
        profits = metric_values["profits"]
        weights = metric_values["weights"]
        times   = metric_values["times"]
        number_city = metric_values["number_city"]
        number_obj  = metric_values["number_obj"]
        # normalization
        #profits = normalization(profits)
        # calculate fitness
        fitness = (profits - R*times) * number_obj
        #summary(profits=profits, weights=weights, times=times, fitness=fitness)
        return fitness
    # TTP =================================

    # TTP ------------------------------
    def fitnessF1scoreTTP(self, metric_values, R=1):
        """
        Fitness cu decadere liniara.
        Pentru fiecare individ:
        - mergem pe traseu (route)
        - cand ajungem intr-un oras luam obiectele de acolo
        - profitul scade liniar cu timpul:
                p(t) = p0 - alpha * t
        - viteza scade in functie de greutatea acumulata
        - costul de timp este penalizat cu R
        Returneaza:
            vector np.array cu fitness pentru fiecare individ
        """
        # unpack metrics
        profits     = metric_values["profits"]
        times       = metric_values["times"]
        weights     = metric_values["weights"]
        number_city = metric_values["number_city"]
        number_obj  = metric_values["number_obj"]
        # normalization
        profits = normalization(profits)
        times   = min_norm(times)
        #mask_city = self.__cityBinaryTSP(number_city)
        # calculate fitness
        #fitness = mask_city * number_obj * ((profits * times) / (profits + R*times+1e-7))
        fitness = number_obj * ((profits * times) / (profits + R*times+1e-7))
        #summary(profits=profits, weights=weights, times=times, fitness=fitness)
        return fitness
    # TTP =================================

def summary(**kw):
    print("Summary")
    for name in kw.keys():
        val = kw[name]
        print("{}: min {}, max {}, mean {}, std {}".format(name, val.min(), val.max(), np.mean(val), np.std(val)))

def normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_ret = (x_max-x)/(x_max-x_min+1e-7)
    return x_ret

def min_norm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 0.1
        x[:] = 0.1
    return (2*x_min)/(x+x_min)
