#!/usr/bin/python

import numpy as np
from root_GA import *

class TSP(RootGA, InitPopulation, Selection, Mutate, Crossover):
    def __init__(self, name="", **configs):
        super().__init__(name)
        Mutate.__init__(**configs)
        
        self.__distance_evolution = np.zeros(5, dtype=np.float32)
        self.__prev_best    = 0
        self.__min_distance = 0

    def __str__(self):
        return str(RootGA)

    def __set_config(self, configs):
        self.mutate = Mutate()

    def __call__(self, map_distances:"np.array", population=None):
        """
        Populatia este compusa din indivizi ce au un fitnes mai slab si elita care are cel mai mare fitness.
        Indivizii sunt compusi din alele, (o alela este un numar intreg 0..GENOME_LENGTH)
        Numarul de alele este GENOME_LENGTH + 1
        Numarul populatiei totale este 'POPULATION_SIZE', numarul elitei este 'ELITE_SIZE'.
        Indivizii care alcatuiesc elita sunt pusi in coada populatiei, pentru a face posibil ca unii indivizi din elita sa se incruciseze si cu indivizi din populatia obisnuita.
        Indivizii care fac parte din elita pot avea un numar mai mare de parteneri, dar un numar mic de copii pentru a evita cazuri de minim local.
        Indivizii din populatia simpla au numar mai mic de parteneri dar un numar mai mare de copii, pentru a diversifica populatia.
        map_distances - distanta dintre orase
        population    - populatia lista de indivizi
        """
        # save map
        self.__runChecks(map_distances)
        # initiaizarea populatiei
        if (population is None):
            population = self.initPopulation(TSP.POPULATION_SIZE)
            #print("population", population)
            self.initFuncParameters(population)
        # init fitness value
        fitness_values = self.fitness(population)
        # obtinerea pozitiei pentru elite
        args_elite = self.getArgElite(fitness_values)
        # calculeaza, parametri functionali pentru parinte 1: 
        #             numarul de parinti 1, 
        #             numarul de parteneri, 
        #             numarul de copii ce poate avea cu un partener, 
        #             numarul de copii ce vor face parte din noua generatie
        size_parent1, nbr_parteners, nbr_childrens, nbr_best_childrens = self.calculateParameterValue()

        # evolutia generatiilor
        for generation in range(TSP.GENERATIONS):
            # nasterea unei noi generatii
            new_population = []
            # selectarea pozitiilor pentru parintii 1
            args_parents1 = self.selectParent1(fitness_values, size_parent1)
            #print("args_parents1", np.unique(args_parents1, return_counts=True)[1])
            # selectarea parintilor 2, care sunt diferiti de parinti 1
            valid_args_parents2 = self.selectValidPopulation(args_parents1, fitness_values[args_parents1])
            valid_population_parents2 = population[valid_args_parents2]
            #print("valid_population_parents2 {}".format(valid_population_parents2.shape))
            valid_fitness_parents2    = fitness_values[valid_args_parents2]
            # bucla completarea noii generatii cu cei mai buni copii
            for arg_parent1, nbr_partener, nbr_children, nbr_best_childs in zip(args_parents1, nbr_parteners, nbr_childrens, nbr_best_childrens):
                p1_childrens  = []
                # selectarea unui numar de parteneri pentru parintele 1
                args_parents2 = self.selectParents2(valid_fitness_parents2, fitness_values[arg_parent1], nbr_partener)
                parent1       = population[arg_parent1]
                # bucla nastere copii, parinte 1 cu toate partenerele
                for arg_parent2 in args_parents2:
                    parent2   = valid_population_parents2[arg_parent2]
                    # incrucisarea parintilor
                    # nasterea copiilor pentru parintele 1 cu una din partenere
                    childrens = self.crossover(parent1, parent2, nbr_children)
                    # mutatii
                    childrens = self.mutate(childrens, parent1, parent2) # in_place operation
                    # adauga noii copii, la numarul total de copii a parinte 1 cu toate partenerele sale
                    p1_childrens.append(childrens)
                # concatenarea tuturor copiilor pentru parintele 1
                p1_childrens = np.concatenate(p1_childrens, axis=0)
                # calcularea fitnesului pentru copii parintelui 1
                childs_fitness_values = self.childFitness(p1_childrens)
                #print("childrens {}, childs_fitness_values {}, shape {}".format(childrens, childs_fitness_values, childrens.shape))
                #print("childrens shape {}, nbr part {}, nbr_child {}, nbr_best_child {}".format(p1_childrens.shape, nbr_partener, nbr_children, nbr_k_best_childs))
                # selectarea copiilor pentru parintele 1
                arg_childrens = self.getArgBestChild(childs_fitness_values, nbr_best_childs)
                # in noua populatie sunt selectati copii cu cel mai bun fitness
                new_population.append(p1_childrens[arg_childrens])
            # salvarea indivizilor ce fac parte din elita
            elite_individs = population[args_elite]
            # integrarea indivizilor din elita in noua generatie, pe ultimele pozitii (in coada listei populatiei)!!!!!
            new_population.append(elite_individs)
            # schimbarea generatiei
            population = np.concatenate(new_population, axis=0)
            # calculare fitness
            fitness_values = self.fitness(population)
            # obtinerea pozitiei pentru elite
            args_elite     = self.getArgElite(fitness_values)
            # calculare metrici
            best_individ, best_fitness, best_distance, best_number_city, distance, number_city = self.clcMetrics(population, fitness_values)
            self.evolutionMonitor(best_distance)
            #self.log(population, fitness_values, args_elite, elite_individs, best_distance)
            # adaugare stres in populatie atunci cand lipseste progresul
            fitness_values = self.stres(population, fitness_values, best_individ, best_distance)
            # afisare metrici
            self.showMetrics(generation, best_individ, best_fitness, best_distance, best_number_city, distance, number_city)

        return best_individ, population

    def calculateParameterValue(self):
        """Calculeaza:
        - numarul de parinti 1,
        - numarul de parteneri ce are parinte 1
        - numarul de copii care ii poate avea parinte 1 cu parinte 2
        - numarul de copii final ce ii poate avea parinte 1
        """
        # calculeaza populatia din noua generatie, inclusiv elita
        population_size_out_extreme = TSP.POPULATION_SIZE-TSP.ELITE_SIZE
        # numarul de parinti 1 care vor fi selectati
        size_parent1 = int(population_size_out_extreme/2.2)
        # calculeaza numarul de parteneri pentru fiecare parinte 1,
        nbr_parteners = np.zeros(size_parent1, dtype=np.int32)+7
        nbr_parteners[-TSP.ELITE_SIZE:] = 150
        # calculeaza numarul de copii care poate sa ii aiba parinte 1
        nbr_childrens = np.zeros(size_parent1, dtype=np.int32)+3
        nbr_childrens[-TSP.ELITE_SIZE:] = 2
        # din numarul total de copii care poate sa ii aiba parinte 1, selecteaza cei mai buni
        nbr_best_childrens = np.ones(size_parent1, dtype=np.int32) # parintii care fac parte din elita pot avea un singur copil
        # calculeaza numarul de copii care ii poate avea, un individ simplu
        k_best_childrens   = (population_size_out_extreme-TSP.ELITE_SIZE)//(size_parent1-TSP.ELITE_SIZE)
        # salveaza numarul de copii
        nbr_best_childrens[:-TSP.ELITE_SIZE] = k_best_childrens
        # calculeaza cati copii, trebuie de adaugat pentru a avea o populatie deplina
        # 
        tmp_big = population_size_out_extreme-(size_parent1-TSP.ELITE_SIZE)*k_best_childrens-TSP.ELITE_SIZE
        # adauga pentru primii indivizi parinte 1, cate un copil pentru a avea o populatie deplina
        nbr_best_childrens[:tmp_big] = k_best_childrens+1
        # afisare parametri 
        #print("nbr_best_childrens", nbr_best_childrens)
        #print("nbr_parteners", nbr_parteners)
        #print("nbr_childrens", nbr_childrens)
        return size_parent1, nbr_parteners, nbr_childrens, nbr_best_childrens

    def evolutionMonitor(self, best_distance):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        best_distance - cea mai buna distanta
        """
        # monitorizare distanta
        self.__distance_evolution[:-1] = self.__distance_evolution[1:]
        self.__distance_evolution[-1]  = best_distance

    def setElites(self, population, elites):
        if (population is None):
            population = self.initPopulation(TSP.POPULATION_SIZE)
        fitness_values = self.fitness(population)
        args = self.getArgWeaks(fitness_values, elites.shape[0])
        population[args] = elites
        return population

    def log(self, population, fitness_values, args_elite, elite_individs, best_distance):
        if (self.__prev_best < best_distance):
            args_ones = np.argwhere(fitness_values[args_elite]==1).reshape(-1)
            print("extreme fittness {}".format(fitness_values[args_elite]))
            print("test extreme fittness {}".format(self.childFitness(population[args_elite])))
            for arg in args_ones:
                best_individ  = population[args_elite][arg]
                best_distance = self.getIndividDistance(best_individ)
                print("best_distance {}".format(best_distance))
                
            print("prev extreme fittness {}".format(self.childFitness(elite_individs)))
            print("prev extreme distance {}".format(self.getDistances(elite_individs)))

        self.__prev_best = best_distance

    def clcMetrics(self, population, fitness_values):
        """
        Calculare metrici:
            population - populatia compusa dintr-o lista de indivizi
            fitness_values - valorile fitnes pentru fiecare individ
        """
        # obtinerea celui mai bun individ
        arg_best = self.getArgBest(fitness_values)
        # selectarea celei mai bune rute
        best_individ  = population[arg_best]
        best_fitness  = fitness_values[arg_best]
        best_distance = self.getIndividDistance(best_individ)
        best_number_city = self.getIndividNumberCities(best_individ)
        # selectarea celei mai mici distante din intreaga populatie
        distance    = self.getBestDistance(population)
        # selectarea celui mai mare numar de orase din intreaga populatie
        number_city = self.getBestNumberCities(population)

        return best_individ, best_fitness, best_distance, best_number_city, distance, number_city

    def showMetrics(self, generation, best_individ, best_fitness, best_distance, best_number_city, distance, number_city):
        """Afisare metrici"""
        #metric_info ="""Generatia: {}, Distanta: {:.3f}, Numarul oraselor {}, Best fitness {:.3f}, Best Distanta: {:.3f}, Best Numarul oraselor {}, Min distance {:.3f}""".format(
        #                            generation, distance, number_city, best_fitness, best_distance, best_number_city, self.__min_distance)
        metric_info ="""{}, Generatia: {}, Distanta: {:.3f}, Best fitness {:.3f}, Distanta min: {:.3f}, Min distance {:.3f}""".format(self.__name,
            generation, best_distance, best_fitness, distance, self.__min_distance)
        print(metric_info)

    def __runChecks(self, distance):
        if (distance is not None):
            self.distance = distance
            # update numarul de orase
            TSP.GENOME_LENGTH = self.distance.shape[0]
        else:
            raise NameError("Parametrul 'distance' este o valoare 'None'")

    def setDataset(self, distance):
        if (distance is not None):
            self.distance = distance
            # update numarul de orase
            TSP.GENOME_LENGTH = self.distance.shape[0]
        else:
            raise NameError("Parametrul 'distance' este o valoare 'None'")

    def initFuncParameters(self, population):
        """Initializare perametri functionali"""
        #self.__min_distance = self.getBestDistance(population)

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)
        new_individ = np.concatenate((TSP.GENOME_LENGTH-1, new_individ, TSP.GENOME_LENGTH-1), axis=None)
        return new_individ

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = TSP.POPULATION_SIZE
        size = (population_size, TSP.GENOME_LENGTH-1)
        population = np.arange(np.prod(size), dtype=np.int32).reshape(*size)%(TSP.GENOME_LENGTH-1)
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=population)
        print("population {}".format(population.shape))
        return population

    def individReconstruction(self, individ):# TO DO: aplica shift sau permutare pe secvente mai mici
        """Initializare individ, cu drumuri aleatorii si oras de start
        start_gene - orasul de start
        """
        cond = np.random.randint(low=0, high=2, size=None)
        size_shift = np.random.randint(low=1, high=TSP.GENOME_LENGTH-6, size=None)
        if (cond == 0):
            individ[1:-1] = np.roll(individ[1:-1], size_shift)
        else:
            args = np.random.choice(5, size=5, p=None)+size_shift
            individ[size_shift:size_shift+5] = individ[args]
        return individ

    def genomeGroupsIndivid(self, individ, individ_T):# TO DO
        """
        Cauta secvente identice de cod, in codul genetic al unui individ,
        individ   - vector compus din codul genetic
        individ_T - individ, cu codul genetic (TSP.GENOME_LENGTH, 1)
        """
        pos_genoms = np.full(individ.shape, individ.shape[0]+1, dtype=np.int32)
        args_pos, args_genoms = np.nonzero(individ == individ_T)
        pos_genoms[args_pos]  = args_genoms
        #print("pos_genoms", pos_genoms)
        diff = np.abs(pos_genoms[:-1] - pos_genoms[1:])
        print("diff", diff)
        return (diff == 1)

    def findSimilarIndivids(self, population, individ, tolerance):
        """
        Cauta indivizi din intreaga populatie ce are codul genetic identic cu un individ,
        population - lista de indivizi
        individ    - vector compus din codul genetic
        tolerance  - cate gene pot fi diferite
        """
        tmp = (population==individ).sum(axis=1)
        return np.argwhere(tmp>=tolerance)

    def similarIndivids(self, population):
        """
        Returneaza un vector de flaguri pentru fiecare individ din populatie daca este gasit codul genetic si la alti indivizi
        population - lista de indivizi
        """
        # initializare vector de flaguri pentru fiecare individ
        similar_args_flag = np.zeros(TSP.POPULATION_SIZE, dtype=bool)
        # setare toleranta, numarul total de gene
        tolerance = TSP.GENOME_LENGTH
        # 
        for i in range(TSP.POPULATION_SIZE-1, -1, -1):
            if (similar_args_flag[i]):
                pass
            else:
                individ = population[i]
                similar_args = self.findSimilarIndivids(population, individ, tolerance)
                #print("similar_args", similar_args)
                similar_args_flag[similar_args] = True
                similar_args_flag[i] = False # scoate flagul de pe individul care este copiat
        return similar_args_flag

    def permuteSimilarIndivids(self, population):
        """
        Returneaza un vector de flaguri pentru fiecare individ din populatie daca este gasit codul genetic si la alti indivizi
        population - lista de indivizi
        """
        similar_args_flag = self.similarIndivids(population)
        print("Similar individs: size {}, population size: {}".format(similar_args_flag.sum(), TSP.POPULATION_SIZE))
        for i in range(TSP.POPULATION_SIZE):
            if (similar_args_flag[i]):
                population[i] = self.individReconstruction(population[i])

    def stres(self, population, fitness_values, best_individ, best_distance):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population    - populatia
        best_individ  - individul cu cel mai bun fitness
        best_distance - cea mai buna distanta
        """
        check_distance = np.allclose(self.__distance_evolution.mean(), best_distance, rtol=1e-03, atol=1e-08)
        #print("distance evolution {}, distance {}".format(check_distance, best_distance))
        if (check_distance):
            self.__distance_evolution[:] = 0
            TSP.MUTATION_RATE = 0.5
            self.permuteSimilarIndivids(population)
            fitness_values = self.fitness(population)
        else:
            TSP.MUTATION_RATE *= 0.9
        return fitness_values

    def getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        distances = self.distance[individ[:-1], individ[1:]]
        distance  = distances.sum() + self.distance[individ[-1], individ[0]]
        return distance

    def getIndividNumberCities(self, individ):
        return np.unique(individ[:-1], return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestDistance(self, population):
        """Calculeaza cea ma buna distanta, din intreaga populatie"""
        distances = self.getDistances(population)
        return distances.min()

    def getBestNumberCities(self, population):
        """Calculeaza cel mai mare numar de orase din intreaga populatie"""
        number_city = self.getNumberCities(population)
        return number_city.max()

    def getArgBestChild(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness pentru un copil
        size           - numarul de argumente cu cei mai buni copii
        """
        args = np.argpartition(fitness_values,-size)
        return args[-size:]

    def getArgWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        args = np.argpartition(fitness_values, size)
        return args[:size]

    def getArgElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        args = np.argpartition(fitness_values,-TSP.ELITE_SIZE)
        args = args[-TSP.ELITE_SIZE:]
        return args

    def getDistances(self, population):
        """Calculaza distanta pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.getIndividDistance,
                                        axis=1,
                                        arr=population)

    def getNumberCities(self, population):
        """Calculeaza numarul de orase pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.getIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def fitness(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        # calculeaza distanta
        distances   = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # normalizeaza intervalul 0...1
        number_city = self.cityNorm(number_city)
        distances   = self.distanceNorm(distances)
        #print("distances: min {:.3f}, max {:.3f}, mean {:.3f}, std {:.3f}, quatile_25 {:.3f}, quatile_50 {:.3f}, quatile_75 {:.3f}".format(distances.min(), distances.max(), np.mean(distances), np.std(distances),
        #                                                          np.quantile(distances, 0.25), np.quantile(distances, 0.5), np.quantile(distances, 0.75)))
        #print("number_city", number_city)
        #print("neighbor_city", neighbor_city)
        distances = distances**2
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        return fitness_values

    def distanceNorm(self, distances):
        mask_not_zero   = (distances!=0)
        valid_distances = distances[mask_not_zero]
        if (valid_distances.shape[0] > 0):
            self.__min_distance = valid_distances.min()
        else:
            self.__min_distance = 0.1
        return (2*self.__min_distance)/(distances+self.__min_distance)

    def cityNorm(self, number_city):
        mask_cities = (number_city==TSP.GENOME_LENGTH)
        return mask_cities.astype(np.float32)

    def childFitness(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        # calculeaza distanta
        distances   = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # normalizeaza intervalul 0...1
        number_city = self.cityNormChilds(number_city)
        distances   = self.distanceNormChilds(distances)
        distances   = distances**2
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        return fitness_values

    def distanceNormChilds(self, distances):
        return (2*self.__min_distance)/(distances + self.__min_distance)

    def cityNormChilds(self, number_city):
        mask_cities = (number_city>=(TSP.GENOME_LENGTH-3))
        return mask_cities.astype(np.float32)


    def __init_fn(self, **kw):

