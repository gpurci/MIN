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
        return self.fn(metric_values, **self.__configs)

    def __str__(self):
        info = """Fitness: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __unpack_method(self, method):
        fn = self.fitnessAbstract
        if (method is not None):
            if   (method == "TSP_f1score"):
                fn = self.fitnessF1scoreTSP
            elif (method == "TSP_norm"):
                fn = self.fitnessNormTSP
            elif (method == "TTP"):
                fn = self.fitnessTTP
        return fn

    def help(self):
        info = """Fitness:
    metoda: 'TSP_f1score'; config: None;
    metoda: 'TSP_norm';    config: None;
    metoda: 'TTP';         config: -> "R":1, ;\n"""
        return info

    def __setMethods(self, method):
        self.__method = method
        self.fn = self.__unpack_method(method)

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
        distances   = self.__distanceF1scoreTSP(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        return fitness_values

    def __distanceF1scoreTSP(self, distances):
        mask_not_zero   = (distances!=0)
        valid_distances = distances[mask_not_zero]
        if (valid_distances.shape[0] > 0):
            min_distance = valid_distances.min()
        else:
            min_distance = 0.1
            distances[:] = 0.1
        return (2*min_distance)/(distances+min_distance)

    def __cityBinaryTSP(self, number_city):
        mask_cities = (number_city>=(self.GENOME_LENGTH-1)).astype(np.float32)
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
        #print("number_city {}".format(number_city))
        number_city = self.__cityNormTSP(number_city)
        #print("number_city {}".format(number_city.sum()))
        distances   = self.__distanceNormTSP(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        #print("fitness {}".format(fitness_values))
        return fitness_values

    def __distanceNormTSP(self, distances):
        min_distance = distances.min()
        max_distance = distances.max()
        return (max_distance-distances)/(max_distance-min_distance)

    def __cityNormTSP(self, number_city):
        mask_cities = (number_city>=(self.GENOME_LENGTH-5)).astype(np.float32)
        return mask_cities*(number_city/self.GENOME_LENGTH)**5
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
        times   = metric_values["times"]
        # calculate fitness
        fitness = profits - R*times
        return fitness
    # TTP =================================
