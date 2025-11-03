#!/usr/bin/python

import numpy as np

class GroupTSP(object):
    # constante pentru setarea algoritmului
    POPULATION_SIZE = 100 # numarul populatiei
    GENOME_LENGTH   = 4 # numarul de orase
    MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
    CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
    SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
    K_DISTANCE      = 0.1   # coeficientul de pondere a distantei si inhibare a numarului de orase
    K_NBR_CITY      = 0.1   # coeficientul de pondere a distantei si inhibare a numarului de orase
    K_BEST          = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor
    K_WRONG         = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mic scor
    GENERATIONS     = 500 # numarul de generatii

    def __init__(self, evolution_algoritms):
        self.nbr_groups = len(evolution_algoritms)
        self.groups     = evolution_algoritms
        self.populations   = [None for _ in range(self.nbr_groups)]
        self.best_individs = [None for _ in range(self.nbr_groups)]
        self.setGroupParameters(GENERATIONS=5)
        print(self)
        self.__distance_evolution = np.zeros(5, dtype=np.float32)

    def __call__(self, map_distance:"np.array"):
        self.groups[-1].setDataset(map_distance)
        # evolutia generatiilor
        for generation in range(GroupTSP.GENERATIONS):
            for i in range(self.nbr_groups-1):
                print("Idx", i)
                population = self.populations[i]
                best_individ, population = self.groups[i](map_distance, population)
                self.best_individs[i] = best_individ
                self.populations[i]   = population
            # 
            print("Last group")
            migrants   = [migrant.reshape(1, -1) for migrant in self.best_individs[:-1]]
            migrants   = np.concatenate(migrants, axis=0)
            population = self.populations[-1]
            population = self.groups[-1].setElites(population, migrants)
            best_individ, population = self.groups[-1](map_distance, population)
            self.best_individs[-1] = best_individ
            self.populations[-1]   = population
            # 
            fitness_values = self.groups[-1].fitness(population)
            # calculate metrics
            best_individ, best_distance, best_number_city, best_fitness, distance, number_city = self.groups[-1].clcMetrics(population, fitness_values)
            self.groups[-1].showMetrics(generation, best_individ, best_distance, best_number_city, best_fitness, distance, number_city)

            self.evolutionMonitor(fitness_values, best_individ, best_distance, best_fitness)
            self.migration(best_distance)

        return best_individ

    def __str__(self):
        info = ""
        for group in self.groups:
            info += str(group) +"\n"
        return info

    def setGroupParameters(self, **kw):
        for group in self.groups:
            group.setParameters(**kw)

    def setParameters(self, **kw):
        size_generations = kw.get("GENERATIONS", None)
        if (size_generations is not None):
            GroupTSP.GENERATIONS = size_generations
            del kw["GENERATIONS"]
        print(kw)
        for group in self.groups:
            group.setParameters(**kw)

    def evolutionMonitor(self, fitness_values, best_individ, best_distance, best_fitness):
        self.__distance_evolution[:-1] = self.__distance_evolution[1:]
        self.__distance_evolution[-1]  = best_distance

    def migration(self, best_distance):
        """functia de migrare, se aplica atunci cand ajungem intr-un minim local,
        migreaza cei mai buni indivizi in grupurile vecine
        """
        check_distance = self.__distance_evolution.mean()
        if (np.allclose(check_distance,  best_distance, rtol=1e-03, atol=1e-08)):
            self.__distance_evolution[:] = 0
            #best_individ = self.best_individs[-1]
            for i in range(1, self.nbr_groups-1):
                print("Migration")
                population = self.populations[i]
                migrant    = self.best_individs[i-1].reshape(1, -1)
                population = self.groups[i].setElites(population, migrant)
            else:
                population = self.populations[0]
                migrant    = self.best_individs[-2].reshape(1, -1)
                population = self.groups[0].setElites(population, migrant)

            distances = []
            for i in range(self.nbr_groups-1):
                print("Check weak group")
                best_individ = self.best_individs[i]
                distance     = self.groups[i].getIndividDistance(best_individ)
                distances.append(distance)
            distances = np.array(distances, dtype=np.float32)
            arg_min = np.argmax(distances).reshape(-1)[0]
            print("Schimba cel mai slab group", np.argmin(distances), distances)
            self.populations[arg_min] = self.groups[arg_min].initPopulation()
            self.groups[arg_min].initFuncParameters(population)
