#!/usr/bin/python

import numpy as np
from root_GA import *

class Fitness(RootGA):
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)

    def __call__(self, metric_values):
        return self.__fn(metric_values, **self.__configs)

    def __str__(self):
        info = """Fitness: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __unpackMethod(self, method, extern_fn):
        fn = self.fitnessAbstract
        if (method is not None):
            if   (method == "TSP_f1score"):
                fn = self.fitnessF1scoreTSP
            elif (method == "TSP_norm"):
                fn = self.fitnessNormTSP
            elif (method == "TTP_f1score"):
                fn = self.fitnessF1scoreTTP
            elif (method == "TTP"):
                fn = self.fitnessTTP
            elif ((method == "extern") and (extern_fn is not None)):
                fn = extern_fn

        return fn

    def help(self):
        info = """Fitness:
    metoda: 'TSP_f1score'; config: None;
    metoda: 'TSP_norm';    config: None;
    metoda: 'TTP_f1score'; config: -> "R":1, ;
    metoda: 'TTP';         config: -> "R":1, ;
    metoda: 'extern';      config: 'extern_kw';\n"""
        return info

    def __setMethods(self, method):
        self.__method = method
        self.__extern_fn = self.__configs.pop("extern_fn", None)
        self.__fn = self.__unpackMethod(method, self.__extern_fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if (self.__extern_fn is not None):
            self.__extern_fn.setParameters(**kw)

    def fitnessAbstract(self, metric_values:dict):
        raise NameError("Lipseste metoda '{}',pentru functia de 'Fitness', configs '{}'".format(self.__method, self.__configs))

    # TSP F1score problem------------------------------
    def fitnessF1scoreTSP(self, metric_values:dict):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        metric_values - metricile pentru fiecare individ
        """
        # despacheteaza metricile
        distances   = metric_values["distances"]
        number_city = metric_values["number_city"]
        # normalizeaza intervalul 0...1
        #print("number_city {}".format(number_city))
        number_city = self.__cityBinaryTSP(number_city)
        #print("number_city {}".format(number_city.sum()))
        distances   = min_norm(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        return fitness_values

    def __cityBinaryTSP(self, number_city):
        mask_cities = (number_city>=self.GENOME_LENGTH).astype(np.float32)
        return mask_cities
    # TSP F1score problem=================================

    # TSP Norm problem------------------------------
    def fitnessNormTSP(self, metric_values:dict):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        metric_values - metricile pentru fiecare individ
        """
        # despacheteaza metricile
        distances   = metric_values["distances"]
        number_city = metric_values["number_city"]
        # normalizeaza intervalul 0...1
        number_city = self.__cityBinaryTSP(number_city)
        distances   = normalization(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        #print("fitness {}".format(fitness_values))
        return fitness_values
    # TSP Norm problem=================================

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
        # normalization
        #profits = normalization(profits)
        mask_city = self.__cityBinaryTSP(number_city)
        # calculate fitness
        fitness = (profits - R*times) * mask_city
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
        mask_city = self.__cityBinaryTSP(number_city)
        # calculate fitness
        fitness = mask_city * number_obj * ((profits * times) / (profits + R*times+1e-7))
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
    return (x_max-x)/(x_max-x_min)

def min_norm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 0.1
        x[:] = 0.1
    return (2*x_min)/(x+x_min)
