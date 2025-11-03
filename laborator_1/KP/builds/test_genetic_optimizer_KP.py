#!/usr/bin/python
import numpy as np

class TestKP(object):
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
        print("+++start")
        result = self.ga_obj.initPopulation(self.ga_obj.POPULATION_SIZE)
        check = result[:, 0] == result[:, -1]
        if (check.sum() < self.ga_obj.POPULATION_SIZE):
            raise NameError("Primul si ultimul '{}' oras nu coincide '{}'".format(result[:, 0], result[:, -1]))
        print("result {}".format(result))

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
        population     = self.ga_obj.initPopulation(self.ga_obj.POPULATION_SIZE)
        fitness_values = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)

        result = self.ga_obj.selectParent1(population, fitness_values, 3)
        print("fitness_values {}".format(fitness_values))
        print("fitness_values {}".format(np.argsort(fitness_values)))
        print("result {}".format(result))

    def selectParent2(self):
        fitness_values   = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)
        fitness_partener = np.random.uniform(low=0, high=1, size=None)
        print("fitness_values {}".format(fitness_values))
        print("argsort fitness_values {}".format(np.argsort(fitness_values)))

        print("fitness_partener {}".format(fitness_partener))
        for _ in range(20):
            result = self.ga_obj.selectParent2(fitness_values, fitness_partener, 5)
            print("result {}".format(result))

    def crossover(self):
        population = self.ga_obj.initPopulation(2)
        parent1 = population[0]
        parent2 = population[1]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))
        for _ in range(10):
            childs = self.ga_obj.crossover(parent1, parent2, nbr_childs=5)
            print("childs {}".format(childs))

    def mutate(self):
        population = self.ga_obj.initPopulation(2)
        parent1 = population[0]
        parent2 = population[1]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))
        childrens = self.ga_obj.crossover(parent1, parent2, nbr_childs=5)
        #print("childrens {}".format(childrens))
        # 
        self.ga_obj.setParameters(
            MUTATION_RATE = 0.5  # threshold-ul pentru a face o mutatie genetica
            )

        for _ in range(10):
            mutate_childrens = self.ga_obj.mutate(childrens.copy())
            tmp_cmp = mutate_childrens!=childrens
            if (tmp_cmp.sum() != 0):
                print("Operatia de mutatie a fost aplicata: {}".format(tmp_cmp))
            else:
                print("Operatia de mutatie 'lipseste'")

    def getIndividDistance(self):
        """Calculul distantei rutelor"""
        population = self.ga_obj.initPopulation(1)
        individ    = population[0]
        print("individ {}".format(individ))
        distance = self.ga_obj.getIndividDistance(individ)
        print("distance {}".format(distance))

    def getIndividNumberCities(self):
        population = self.ga_obj.initPopulation(1)
        individ    = population[0]
        print("individ {}".format(individ))
        number_city = self.ga_obj.getIndividNumberCities(individ)
        print("number_city {}".format(number_city))

    def getDistances(self):
        """calcularea distantei pentru fiecare individ din populatiei"""
        population = self.ga_obj.initPopulation(self.ga_obj.POPULATION_SIZE)
        print("population {}".format(population))
        distances = self.ga_obj.getDistances(population)
        print("distances {}".format(distances))

    def getNumberCities(self):
        # calculeaza numarul de orase unice
        population = self.ga_obj.initPopulation(self.ga_obj.POPULATION_SIZE)
        print("population {}".format(population))
        number_city = self.ga_obj.getNumberCities(population)
        print("GENOME_LENGTH {}".format(self.ga_obj.GENOME_LENGTH))
        print("number_city {}".format(number_city))

    def distanceNorm(self):
        population = self.ga_obj.initPopulation(self.ga_obj.POPULATION_SIZE)
        self.ga_obj.initFuncParameters(population)
        print("population {}".format(population))
        distances = self.ga_obj.getDistances(population)
        print("distances {}".format(distances))
        print("arg distance {}".format(np.argsort(distances)[::-1]))
        distances = self.ga_obj.distanceNorm(distances)
        print("norm distances {}".format(distances))
        print("arg distance {}".format(np.argsort(distances)))

    def getKBest(self):
        # returneaza pozitiile cu cele mai mari valori
        population = self.ga_obj.initPopulation(2)
        self.ga_obj.initFuncParameters(population)
        parent1 = population[0]
        parent2 = population[1]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))
        childrens = self.ga_obj.crossover(parent1, parent2, 25)
        fitness_values = self.ga_obj.fitness(childrens)
        print("fitness_values {}".format(fitness_values))
        print("argsort fitness_values {}".format(np.argsort(fitness_values)))
        k_best = self.ga_obj.getKBest(fitness_values, 5)
        print("k_best {}".format(k_best))


    def getKExtreme(self):
        fitness_values = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)
        print("fitness_values {}".format(fitness_values))
        arg_sort = np.argsort(fitness_values)
        print("sort fitness_values {}".format(fitness_values[arg_sort]))
        print("arg sort fitness_values {}".format(arg_sort))
        arg_extreme = self.ga_obj.getKExtreme(fitness_values)
        print("arg_extreme {}".format(arg_extreme))
        print("extreme fitness {}".format(fitness_values[arg_extreme]))

    def getKWeaks(self):
        fitness_values = np.random.uniform(low=0, high=1, size=self.ga_obj.POPULATION_SIZE)
        print("fitness_values {}".format(fitness_values))
        arg_sort = np.argsort(fitness_values)[::-1]
        print("sort fitness_values {}".format(fitness_values[arg_sort]))
        print("arg sort fitness_values {}".format(arg_sort))
        arg_extreme = self.ga_obj.getKWeaks(fitness_values, 10)
        print("arg_extreme {}".format(arg_extreme))
        print("extreme fitness {}".format(fitness_values[arg_extreme]))

    def permuteSimilarIndivids(self):
        a = np.arange(40, dtype=np.int32).reshape(4, 10)%10
        print(a)
        a = np.apply_along_axis(np.random.permutation,
                                                axis=1,
                                                arr=a)
        print(a)
        population = np.repeat(a, [1, 2, 3, 4], axis=0)
        population = np.random.permutation(population)
        self.ga_obj.setParameters(
            POPULATION_SIZE = population.shape[0],  # numarul populatiei
            )
        print("population shape {}".format(population.shape))
        # 
        similar_arg_flag = self.ga_obj.similarIndivids(population)
        for flag, individ in zip(similar_arg_flag, population):
            print(flag, individ)
        ##
        self.ga_obj.permuteSimilarIndivids(population)
        act_similar_arg_flag = self.ga_obj.similarIndivids(population)
        ## 
        print("actual flag:   prev flag:    individ")
        for act_flag, flag, individ in zip(act_similar_arg_flag, similar_arg_flag, population):
            print(act_flag, flag, individ)
        ##
