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

    def __init__(self, config):
        super().__init__()
        self.setConfig(config)
    
    def setMetrics(self, metrics):
        self.metrics = metrics

    def __call__(self, population, metric_values):
        return self.fn(population, metric_values)

    def __config_fn(self):
        self.fn = self.fitnessAbstract
        if self.__config is not None:
            if self.__config == "TSP_f1score":
                self.fn = self.fitnessF1scoreTSP
            elif self.__config == "TTP_linear":
                self.fn = self.fitness_ttp_linear
            elif self.__config == "TTP_exp":
                self.fn = self.fitness_ttp_exp

    def help(self):
        info = """Fitness:
        metode de config: 'TSP_f1score', 'TTP_linear', 'TTP_exp'
        """
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def fitnessAbstract(self, population):
        raise NameError("Lipseste configuratia pentru functia de 'Fitness': config '{}'".format(self.__config))

    # TSP problem------------------------------
    def fitnessF1scoreTSP(self, population, metric_values):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
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
        mask_not_zero = (distances != 0)
        valid_distances = distances[mask_not_zero]
        if valid_distances.shape[0] > 0:
            self.__min_distance = valid_distances.min()
        else:
            self.__min_distance = 0.1
            distances[:] = 0.1
        return (2 * self.__min_distance) / (distances + self.__min_distance)

    def __cityNormTSP(self, number_city):
        mask_cities = (number_city>=(self.GENOME_LENGTH-5)).astype(np.float32)

        return mask_cities
    # TSP problem=================================

    
    # functia fitness cu decadere liniara
    def fitness_ttp_linear(self, population, metric_values):
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

        n = population.shape[0]
        fitness = np.zeros(n, dtype=float)

        for r, route in enumerate(population):

            Wcur = 0.0
            Tcur = 0.0
            Pcur = 0.0

            # vizităm secvenţial
            for i in range(len(route)-1):
                c = route[i]

                # ia items din oraş
                for (city, w, p) in self.items:
                    if city == c:
                        Pcur += max(0.0, p - self.alpha*Tcur)
                        Wcur += w

                v = self.v_max - (self.v_max-self.v_min)*(Wcur/self.W)
                Tcur += self.distance[ c, route[i+1] ] / v

            # întoarcere
            v = self.v_max - (self.v_max-self.v_min)*(Wcur/self.W)
            Tcur += self.distance[ route[-1], route[0] ] / v

            fitness[r] = Pcur - self.R*Tcur

        return fitness



    # functia fitness cu decadere exponentiala
    def fitness_ttp_exp(self, population, metric_values):
        """
        - pe măsură ce vizităm oraşele, luăm obiectele găsite acolo
        - fiecare obiect are profitul iniţial p0
        - dar profitul scade cu timpul deoarece obiectul este „mai puţin valoros” dacă ajungi târziu
        - decădere exponenţială:
            p(t) = p0 * exp(- λ * timp)
        - viteza berlinei scade pe măsură ce rucsacul se încarcă cu obiecte
            v = v_max - (v_max - v_min) * (greutate_curentă / W)

        Returneaza
            fitness = profit_total - R * timp_total
            """

        n = population.shape[0]
        fitness = np.zeros(n, dtype=float)

        for r, route in enumerate(population):

            Wcur = 0.0
            Tcur = 0.0
            Pcur = 0.0

            # mergem secvenţial prin oraşe
            for i in range(len(route)-1):
                c = route[i]

                # luăm iteme din oraş
                for (city, w, p0) in self.items:
                    if city == c:
                        p = p0 * np.exp(-self.lam * Tcur)
                        Pcur += p
                        Wcur += w

                # viteza berlinei
                v = self.v_max - (self.v_max - self.v_min)*(Wcur/self.W)

                # timp până la următorul
                Tcur += self.distance[ c, route[i+1] ] / v

            # închidere ciclu
            v = self.v_max - (self.v_max - self.v_min)*(Wcur/self.W)
            Tcur += self.distance[ route[-1], route[0] ] / v

            fitness[r] = Pcur - self.R*Tcur

        return fitness

    def setTTPParams(self, distance, items, v_min, v_max, W, R, lam=0.01, alpha=0.01):
        self.distance = distance
        self.items    = items
        self.v_min    = v_min
        self.v_max    = v_max
        self.W        = W
        self.R        = R
        self.lam      = lam
        self.alpha    = alpha


