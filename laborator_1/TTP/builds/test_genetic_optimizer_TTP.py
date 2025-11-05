#!/usr/bin/python

import numpy as np
import os



class TestGaoTTP(object):
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

    def __init__(self, genetic_algoritm_object):
        self.ga_obj = genetic_algoritm_object

    def initPopulation(self):

        base = os.path.dirname(__file__)
        nodes = os.path.join(base, "..", "data", "NODE_COORD_SECTION.csv")
        items = os.path.join(base, "..", "data", "ITEMS_SECTION.csv")

        # testăm pe ga_obj
        pop = self.ga_obj.initPopulationBeamTTP(
            nodes_csv_path = nodes,
            items_csv_path = items,
            population_target = 200,
            beam_width = 8,
            lambda_time = 1.0,
            seed = 123,
            allow_pick_in_start_city = False
        )

        print("pop shape:", pop.shape)
        
        # 1) primul = ultimul
        if not np.all(pop[:,0] == pop[:,-1]):
            raise NameError("Primul si ultimul oras nu sunt egale")

        # 2) fiecare element e valid index
        if pop.min() < 0 or pop.max() >= self.ga_obj.GENOME_LENGTH:
            raise NameError("Index invalid in populatie")

        # 3) nu sunt duplicate
        if len(set(map(tuple, pop))) != pop.shape[0]:
            raise NameError("Exista rute duplicate")

        print("Beam init test OK")
        
    def selectValidPopulation(self):
        arg_parents1 = np.random.randint(low=0, high=self.ga_obj.POPULATION_SIZE, size=5)
        fitness_values_parents1 = np.random.uniform(low=0, high=1, size=5)
        print("arg_parents1 {}".format(arg_parents1))
        print("fitness_values_parents1 {}".format(fitness_values_parents1))
        result = self.ga_obj.selectValidPopulation(arg_parents1, fitness_values_parents1)
        print("result {}".format(result))
        if ((result.ndim != 1)):
            raise NameError("Dimensiunea valorilor generate '{}' este diferita de 1".format(result.ndim))
        if ((result.max() >= self.ga_obj.POPULATION_SIZE) or (result.max() < 0)):
            raise NameError("Valorilor generate {}, depasesc numarul populatiei '{}' este diferita de 1".format((result.max(), result.min()), self.ga_obj.POPULATION_SIZE))

    def selectParent1(self):
        population     = self.ga_obj.initPopulation(0)
        fitness_values = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)

        result = self.ga_obj.selectParent1(population, fitness_values, 3)
        print("result {}".format(result))

    def selectParent2(self):
        population     = self.ga_obj.initPopulation(0)
        fitness_values = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)
        fitness_partener = np.random.uniform(low=0, high=1, size=None)
        for _ in range(20):
            result = self.ga_obj.selectParent2(population, fitness_values, fitness_partener)
            print("result {}".format(result))

    def crossover(self):
        population = self.ga_obj.initPopulation(0)
        parent1 = population[0]
        parent2 = population[1]
        for _ in range(20):
            child = self.ga_obj.crossover(parent1, parent2)
            print("child {}".format(child))

    def mutate(self):
        population = self.ga_obj.initPopulation(0)
        individ = population[0]
        for _ in range(20):
            child = self.ga_obj.mutate(individ)
            print("child {}".format(child))
