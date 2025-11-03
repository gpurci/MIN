#!/usr/bin/python

import numpy as np

class TSP(object):
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

    def __init__(self, name=""):
        self.__name = name
        self.__distance_evolution = np.zeros(5, dtype=np.float32)
        self.__hibrid = 0
        self.__prev_best = 0
        self.__min_distance = 0

    def __str__(self):
        info = """name: {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    K_BEST          = {}
    GENERATIONS     = {}""".format(self.__name, TSP.POPULATION_SIZE, TSP.GENOME_LENGTH, TSP.MUTATION_RATE, 
                                    TSP.CROSSOVER_RATE, TSP.SELECT_RATE, TSP.K_BEST, TSP.GENERATIONS)
        return info

    def __call__(self, distance:"np.array", population=None):
        # save map
        self.__runChecks(distance)
        # initiaizarea populatiei
        if (population is None):
            population = self.initPopulation(TSP.POPULATION_SIZE)
            #print("population", population)
            self.initFuncParameters(population)
        # init fitness value
        fitness_values = self.fitness(population)
        # obtinerea pozitiei pentru indivizii extrimali
        arg_extreme = self.getKExtreme(fitness_values)

        population_size_out_extreme = TSP.POPULATION_SIZE-arg_extreme.shape[0]
        ##############
        size_parent1 = int(population_size_out_extreme/2.2)
        #nbr_k_best_childrens = np.random.randint(low=2, high=5, size=size_parent1)
        nbr_k_best_childrens = np.ones(size_parent1, dtype=np.int32)
        #nbr_k_best_childrens[:TSP.K_BEST] = 1
        k_best_childrens = (population_size_out_extreme-TSP.K_BEST)//(size_parent1-TSP.K_BEST)
        nbr_k_best_childrens[:-TSP.K_BEST] = k_best_childrens
        tmp_big = population_size_out_extreme-(size_parent1-TSP.K_BEST)*k_best_childrens-TSP.K_BEST
        nbr_k_best_childrens[:tmp_big] = k_best_childrens+1
        ##################
        nbr_parteners = np.zeros(size_parent1, dtype=np.int32)+7
        nbr_parteners[-TSP.K_BEST:] = 150
        ##################
        nbr_childrens = np.zeros(size_parent1, dtype=np.int32)+3
        nbr_childrens[-TSP.K_BEST:] = 2
        #print("nbr_k_best_childrens", nbr_k_best_childrens)
        #print("nbr_parteners", nbr_parteners)
        #print("nbr_childrens", nbr_childrens)

        # evolutia generatiilor
        for generation in range(TSP.GENERATIONS):
            # nasterea unei noi generatii
            new_population = []
            # selectarea unui parinte
            arg_parents1 = self.selectParent1(fitness_values, arg_extreme, size_parent1)
            #print("arg_parents1", np.unique(arg_parents1, return_counts=True)[1])
            valid_arg_parents2 = self.selectValidPopulation(arg_parents1, fitness_values[arg_parents1])
            valid_population_parents2 = population[valid_arg_parents2]
            #print("valid_population_parents2 {}".format(valid_population_parents2.shape))
            valid_fitness_parents2    = fitness_values[valid_arg_parents2]
            for arg_parent1, nbr_k_best_childs, nbr_partener, nbr_children in zip(arg_parents1, nbr_k_best_childrens, nbr_parteners, nbr_childrens):
                # selectarea celui de al doilea parinte
                p1_childrens = []
                arg_parents2 = self.selectParent2(valid_fitness_parents2, fitness_values[arg_parent1], nbr_partener)
                parent1      = population[arg_parent1]
                # bucla nastere copii, parinte 1 cu toate partenerele
                for arg_parent2 in arg_parents2:
                    parents2  = valid_population_parents2[arg_parent2]
                    # incrucisarea parintilor
                    childrens = self.crossover(parent1, parents2, nbr_children)
                    # mutatii
                    childrens = self.mutate(childrens) # in_place operation
                    p1_childrens.append(childrens)
                # concatenarea tuturor copiilor pentru parintele 1
                p1_childrens = np.concatenate(p1_childrens, axis=0)
                # calcularea fitnesului pentru copii parintelui 1
                childs_fitness_values = self.childFitness(p1_childrens, 2)
                #print("childrens {}, childs_fitness_values {}, shape {}".format(childrens, childs_fitness_values, childrens.shape))
                #print("childrens shape {}, nbr part {}, nbr_child {}, nbr_best_child {}".format(p1_childrens.shape, nbr_partener, nbr_children, nbr_k_best_childs))
                # selectarea copiilor pentru parintele 1
                arg_childrens = self.getKBest(childs_fitness_values, nbr_k_best_childs)
                # in noua populatie sunt selectati copii cu cel mai bun fitness
                new_population.append(p1_childrens[arg_childrens])
            # salvarea indivizilor extrimali
            extreme_population = population[arg_extreme].copy()
            # schimbarea generatiei
            #population = np.array(new_population, dtype=np.int32)
            #new_population.insert(0, extreme_population)
            new_population.append(extreme_population)
            # integrarea indivizilor extrimali in noua generatie
            #print("size old population: {}".format(population.shape))
            population = np.concatenate(new_population, axis=0)
            #print("size new population: {}".format(population.shape))
            # update la valorile fitness
            fitness_values = self.fitness(population)
            # obtinerea pozitiilor pentru indivizii extrimali din noua generatie
            arg_extreme = self.getKExtreme(fitness_values)
            # calculate metrics
            best_individ, best_distance, best_number_city, best_fitness, distance, number_city = self.clcMetrics(population, fitness_values)
            self.evolutionMonitor(population, fitness_values, best_individ, best_distance, best_fitness)
            #self.log(population, fitness_values, arg_extreme, extreme_population, best_distance)
            self.stres(population, best_individ, best_distance)
            self.showMetrics(generation, best_individ, best_distance, best_number_city, best_fitness, distance, number_city)
            fitness_values[-arg_extreme.shape[0]:] *= 0.7

        return best_individ, population

    def evolutionMonitor(self, population, fitness_values, best_individ, best_distance, best_fitness):
        self.__distance_evolution[:-1] = self.__distance_evolution[1:]
        self.__distance_evolution[-1]  = best_distance
        """
                                if (self.__hibrid > 5):
                                    self.__hibrid = 0
                                    self.__distance_evolution[:] = 0
                                    tmp_size_hibrid = TSP.POPULATION_SIZE-(TSP.K_BEST*2)
                                    if (tmp_size_hibrid < 0):
                                        tmp_size_hibrid = 1
                                    new_population = self.initPopulation(tmp_size_hibrid)
                                    population = np.concatenate((new_population, population[tmp_size_hibrid:]), axis=0)
                                    print("Hibrid hh")"""

    def setExtreme(self, population, extremes):
        if (population is None):
            population = self.initPopulation(TSP.POPULATION_SIZE)
        fitness_values = self.fitness(population)
        args = self.getKWeaks(fitness_values, extremes.shape[0])
        population[args] = extremes
        return population

    def setParameters(self, **kw):
        TSP.POPULATION_SIZE = kw.get("POPULATION_SIZE", TSP.POPULATION_SIZE)
        TSP.MUTATION_RATE   = kw.get("MUTATION_RATE", TSP.MUTATION_RATE)
        TSP.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE", TSP.CROSSOVER_RATE)
        TSP.SELECT_RATE     = kw.get("SELECT_RATE", TSP.SELECT_RATE)
        TSP.GENERATIONS     = kw.get("GENERATIONS", TSP.GENERATIONS)
        TSP.K_DISTANCE      = kw.get("K_DISTANCE", TSP.K_DISTANCE)
        TSP.K_NBR_CITY      = kw.get("K_NBR_CITY", TSP.K_NBR_CITY)
        TSP.K_BEST          = kw.get("K_BEST", TSP.K_BEST)
        TSP.K_WRONG         = kw.get("K_WRONG", TSP.K_WRONG)

    def log(self, population, fitness_values, arg_extreme, extreme_population, best_distance):
        if (self.__prev_best < best_distance):
            args_ones = np.argwhere(fitness_values[arg_extreme]==1).reshape(-1)
            print("extreme fittness {}".format(fitness_values[arg_extreme]))
            print("test extreme fittness {}".format(self.childFitness(population[arg_extreme], 2)))
            for arg in args_ones:
                best_individ  = population[arg_extreme][arg]
                best_distance = self.getIndividDistance(best_individ)
                print("best_distance {}".format(best_distance))
                
            print("prev extreme fittness {}".format(self.childFitness(extreme_population, 1)))
            print("prev extreme distance {}".format(self.getDistances(extreme_population)))

        self.__prev_best = best_distance

    def clcMetrics(self, population, fitness_values):
        # obtinerea celui mai bun individ
        arg_best = self.getBestRoute(fitness_values)
        # selectarea celei mai bune rute
        best_individ  = population[arg_best]
        best_fitness  = fitness_values[arg_best]
        best_distance = self.getIndividDistance(best_individ)
        best_number_city = self.getIndividNumberCities(best_individ)
        # selectarea celei mai mici distante din intreaga populatie
        distance = self.getBestDistance(population)
        # selectarea celui mai mare numar de orase din intreaga populatie
        number_city = self.getBestNumberCities(population)

        return best_individ, best_distance, best_number_city, best_fitness, distance, number_city

    def showMetrics(self, generation, best_individ, best_distance, best_number_city, best_fitness, distance, number_city):
        # prezinta metricile
        
        #metric_info ="""Generatia: {}, Distanta: {:.3f}, Numarul oraselor {}, Best fitness {:.3f}, Best Distanta: {:.3f}, Best Numarul oraselor {}, Min distance {:.3f}""".format(
        #                            generation, distance, number_city, best_fitness, best_distance, best_number_city, self.__min_distance)
        metric_info ="""{}, Generatia: {}, Distanta: {:.3f}, Best fitness {:.3f}, Best Distanta: {:.3f}, Min distance {:.3f}""".format(self.__name,
            generation, best_distance, best_fitness, distance, self.__min_distance)
        print(metric_info)

    def __runChecks(self, distance):
        if (distance is not None):
            self.distance = distance
            # update numarul de orase
            TSP.GENOME_LENGTH = self.distance.shape[0]
        else:
            raise NameError("Parametrul 'distance' este o valoare 'None'")

    def setMap(self, distance):
        if (distance is not None):
            self.distance = distance
            # update numarul de orase
            TSP.GENOME_LENGTH = self.distance.shape[0]
        else:
            raise NameError("Parametrul 'distance' este o valoare 'None'")

    def initFuncParameters(self, population):
        # calculeaza distanta
        self.__min_distance = self.getBestDistance(population)
        self.__hibrid = 0
        self.__distance_evolution[:] = 0

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)
        new_individ = np.concatenate((new_individ, new_individ[0]), axis=None)
        return new_individ

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = TSP.POPULATION_SIZE
        size = (population_size, TSP.GENOME_LENGTH)
        arr = np.arange(np.prod(size), dtype=np.int32).reshape(*size)%TSP.GENOME_LENGTH
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=arr)
        return population

    def individReconstruction(self, start_gene):
        """Initializarea populatiei, cu drumuri aleatorii"""
        genes = np.arange(TSP.GENOME_LENGTH, dtype=np.int32)
        genes = np.delete(genes, start_gene, None)
        genes = np.random.permutation(genes)
        return np.concatenate((start_gene, genes, start_gene), axis=None)

    def selectValidPopulation(self, arg_parents1, fitness_values_parents1):
        # select valid parents for parents2, from list of valid parents1
        # 1/3 from parents1 is valid as a parents2
        # create mask of valid parents2 from valid parents1
        fitness_values_parents1 = fitness_values_parents1.copy()
        fitness_values_parents1 = fitness_values_parents1/fitness_values_parents1.sum()
        arg_valid_id_parents1   = np.random.choice(arg_parents1.shape[0], size=arg_parents1.shape[0]//3, p=fitness_values_parents1)
        # exclude valid parents1 from invalid parents selection
        mask = np.ones(arg_parents1.shape[0], dtype=bool)
        #print("mask", mask.shape)
        mask[arg_valid_id_parents1] = False
        # do invalid parents1 for valid parents2
        arg_invalid_parents1 = arg_parents1[mask]
        arg_populations = np.ones(TSP.POPULATION_SIZE, dtype=bool)
        #print("arg_populations", arg_populations.shape)
        arg_populations[arg_invalid_parents1] = False
        return np.argwhere(arg_populations).reshape(-1)

    def selectParent1(self, fitness_values, args_best, size_parents):
        """selectarea unui parinte aleator din populatie, bazandune pe distributia fitness valorilor"""
        # select random parent
        select_rate = np.random.uniform(low=0, high=1, size=None)
        size_parents -= TSP.K_BEST
        if (select_rate < TSP.SELECT_RATE):
            prob_fitness = fitness_values.copy()
            prob_fitness[args_best] = 0.
            prob_fitness = prob_fitness / prob_fitness.sum()
            #print("selectie parent1 fitness size", prob_fitness.shape)
            args = np.random.choice(TSP.POPULATION_SIZE, size=size_parents, p=prob_fitness)
        else:
            args = np.arange(size_parents, dtype=np.int32)

        args = np.concatenate((args, args_best), axis=0)
        return args

    def selectIndividParent2(self, select_cond, pick):
        """selectarea unui parinte aleator din populatie"""
        if (select_cond == 1): # selectie dupa compatibilitate
            current = 0
            for arg, fitness_value in enumerate(self.__parent_fitness, 0):
                current += fitness_value
                if (current > pick):
                    break
        elif (select_cond == 2): # selectie dupa compatibilitate
            arg = np.random.choice(self.__parent_fitness.shape[0], size=None, p=None)
        else: # selectie aleatorie
            # selecteaza argumentul parintelui 2
            arg = np.random.choice(self.__parent_fitness.shape[0], size=None, p=self.__parent_fitness)
        return arg

    def selectParent2(self, fitness_values, fitness_partener, size_parteners):
        """selectarea unui parinte aleator din populatie"""
        # select condition for all parteners
        p = [TSP.SELECT_RATE/2, 1-TSP.SELECT_RATE, TSP.SELECT_RATE/2]
        select_conds = np.random.choice([0, 1, 2], size=size_parteners, p=p)
        # calculate pick for all parteners
        self.__parent_fitness = fitness_values / fitness_values.sum()
        total_fitness = np.sum(self.__parent_fitness, axis=None)
        picks = np.random.uniform(low=fitness_partener, high=total_fitness, size=size_parteners)

        parent_args = []
        for pick, select_cond in zip(picks, select_conds):
            arg = self.selectIndividParent2(select_cond, pick)
            parent_args.append(arg)
        return np.array(parent_args, dtype=np.int32)

        
    def crossoverIndivid(self, parent1, parent2, pos, childs, coords, crossover_conds):
        """Incrucisarea a doi parinti pentru a crea un urmas
        """
        # creare un copil fara mostenire
        child       = childs[pos]
        # selectarea diapazonului de mostenire
        start, end  = coords[pos]
        cond = crossover_conds[pos]
        #print("coords ", coords[pos], "cond", cond)
        if (start > end):
            start, end = end, start
        if (cond == 0):
            # copierea rutei din primul parinte
            child[start:end] = parent1[start:end]
            # copierea rutei din cel de al doilea parinte
            child[:start] = parent2[:start]
            child[end:]   = parent2[end:]
        elif (cond == 1):
            # copierea rutei din primul parinte
            child[start:end] = parent2[start:end]
            # copierea rutei din cel de al doilea parinte
            child[:start] = parent1[:start]
            child[end:]   = parent1[end:]
        elif (cond == 2):
            # copierea rutei din primul parinte
            child[:start] = parent1[:start]
            child[start:] = parent2[start:]
            child[-1] = child[0]
        elif (cond == 3):
            # copierea rutei din primul parinte
            child[:start] = parent2[:start]
            child[start:] = parent1[start:]
            child[-1] = child[0]
        elif (cond == 4):
            # copierea rutei din primul parinte
            args = np.random.choice(np.arange(1, TSP.GENOME_LENGTH), size=TSP.GENOME_LENGTH//2)
            #print("args", args)
            child[:] = parent2[:]
            child[args] = parent1[args]
        elif (cond == 5):
            # copierea rutei din primul parinte
            args = np.random.choice(np.arange(1, TSP.GENOME_LENGTH), size=TSP.GENOME_LENGTH//2)
            #print("args", args)
            child[:] = parent1[:]
            child[args] = parent2[args]
        elif (cond == 6):
            # copierea rutei din primul parinte
            diff_mask = ((parent1-parent2)!=0)
            diff_mask[0] = False
            diff_mask[-1] = False
            args = np.argwhere(diff_mask).reshape(-1)
            args = args[:args.shape[0]//2]
            #print("args", args)
            child[:] = parent1[:]
            child[args] = parent2[args]
        elif (cond == 7):
            # copierea rutei din primul parinte
            diff_mask = ((parent1-parent2)!=0)
            diff_mask[0] = False
            diff_mask[-1] = False
            args = np.argwhere(diff_mask).reshape(-1)
            args = args[:args.shape[0]//2]
            #print("args", args)
            child[:] = parent2[:]
            child[args] = parent1[args]
        elif (cond == 8):
            # copierea rutei din primul parinte
            diff_mask = ((parent1-parent2)!=0)
            diff_mask[0] = False
            diff_mask[-1] = False
            args = np.argwhere(diff_mask).reshape(-1)
            args = args[args.shape[0]//2:]
            #print("args", args)
            child[:] = parent1[:]
            child[args] = parent2[args]
        elif (cond == 9):
            # copierea rutei din primul parinte
            diff_mask = ((parent1-parent2)!=0)
            diff_mask[0] = False
            diff_mask[-1] = False
            args = np.argwhere(diff_mask).reshape(-1)
            args = args[args.shape[0]//2:]
            #print("args", args)
            child[:] = parent2[:]
            child[args] = parent1[args]


    def crossover(self, parent1, parent2, nbr_childs=1):
        """Incrucisarea a doi parinti pentru a crea un urmas
        """
        # creare un copil fara mostenire
        childs = np.zeros((nbr_childs, TSP.GENOME_LENGTH+1), dtype=np.int32)
        # selectarea diapazonului de mostenire
        coords = np.random.randint(low=1, high=TSP.GENOME_LENGTH+1, size=(nbr_childs, 2))
        # medodele de aplicare a incrucisarii
        # cond 0 -> 
        crossover_conds = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=nbr_childs)
        for pos in range(nbr_childs):
            self.crossoverIndivid(parent1, parent2, pos, childs, coords, crossover_conds)
        return childs

    def mutateIndivid(self, individs, pos, mutate_conds, coords):
        """Mutatia genetica a rutei"""
        # selectarea genomurile care vor fi mutate cu locul
        individ = individs[pos]
        cond    = mutate_conds[pos]
        loc1, loc2 = coords[pos]
        if (cond == 0):
            pass
        elif (cond == 1):
            individ[loc1], individ[loc2] = individ[loc2], individ[loc1]
        elif (cond == 2):
            gene  = np.random.choice(np.arange(TSP.GENOME_LENGTH), size=None)
            individ[loc1] = gene

    def mutate(self, individs):
        """Mutatia genetica a rutei, operatie in_place"""
        # selectarea genomurile care vor fi mutate cu locul
        # aplicarea operatiei de mutatie pentru tot numarul de indivizi
        nbr_individs = individs.shape[0]
        # prababilitatea pentru fiecare metoda de mutatie
        p = [1-TSP.MUTATION_RATE, TSP.MUTATION_RATE/2, TSP.MUTATION_RATE/2]
        # cond 0 -> nu se aplica operatia de mutatie
        # cond 1 -> se aplica operatia de mutatie, metoda swap
        # cond 0 -> se aplica operatia de mutatie, se modifica o singura gena
        mutate_conds = np.random.choice([0, 1, 2], size=nbr_individs, p=p)
        # coordonatele lalele-lor
        coords = np.random.randint(low=1, high=TSP.GENOME_LENGTH+1, size=(nbr_individs, 2))
        # loop pentru fiecare individ
        for pos in range(nbr_individs):
            self.mutateIndivid(individs, pos, mutate_conds, coords)
        return individs

    def genomeGroupsIndivid(self, individ, best_individ_T):
        """"""
        pos_genoms = np.full(individ.shape, individ.shape[0]+1, dtype=np.int32)
        arg_pos, arg_genoms = np.nonzero(individ == best_individ_T)
        pos_genoms[arg_pos] = arg_genoms
        #print("pos_genoms", pos_genoms)
        diff = np.abs(pos_genoms[:-1] - pos_genoms[1:])
        print("diff", diff)
        return (diff == 1)

    def findSimilarIndivids(self, population, individ, tolerance):
        tmp = (population==individ).sum(axis=1)
        return np.argwhere(tmp>=tolerance)

    def similarIndivids(self, population):
        similar_arg_flag = np.zeros(TSP.POPULATION_SIZE, dtype=bool)
        tolerance = TSP.GENOME_LENGTH
        for i in range(TSP.POPULATION_SIZE-1, -1, -1):
            if (similar_arg_flag[i]):
                pass
            else:
                individ = population[i]
                similar_args = self.findSimilarIndivids(population, individ, tolerance)
                #print("similar_args", similar_args)
                similar_arg_flag[similar_args] = True
                similar_arg_flag[i] = False
        return similar_arg_flag

    def permuteSimilarIndivids(self, population):
        similar_arg_flag = self.similarIndivids(population)
        tmp_pos = np.arange(1, TSP.GENOME_LENGTH+1, dtype=np.int32)
        #print("similar_arg_flag", similar_arg_flag)
        print("similar size {}, population size {}".format(similar_arg_flag.sum(), TSP.POPULATION_SIZE))
        for i in range(1, TSP.POPULATION_SIZE, 1):
            if (similar_arg_flag[i]):
                population[i] = self.individReconstruction(population[i][0])


    def stres(self, population, best_individ, best_distance):
        """functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population   - populatia
        best_individ - individul cu cel mai bun fitness
        """
        check_distance = self.__distance_evolution.mean()
        #print("distance evolution {}, distance {}".format(check_distance, best_distance))
        if (np.allclose(check_distance,  best_distance, rtol=1e-02, atol=1e-08)):
            self.__distance_evolution[:] = 0
            TSP.MUTATION_RATE = 0.9
            #self.__hibrid += 1
            """
                                                print("Stres best individ", best_individ.shape)
                                                best_individ_T = best_individ[1:-1].reshape(-1, 1)
                                                mask = np.ones(best_individ_T.shape[0]-1, dtype=bool)
                                                for individ in population:
                                                    tmp_secv_genome = self.genomeGroupsIndivid(individ[1:-1], best_individ_T)
                                                    print(tmp_secv_genome.shape)
                                                    mask &= tmp_secv_genome
                                    
                                                print("mask secvent genome", mask)"""
            self.permuteSimilarIndivids(population)

        else:
            TSP.MUTATION_RATE *= 0.9
            self.__hibrid *= 0.99
            #print("MUTATION_RATE {}".format(TSP.MUTATION_RATE))


    def getIndividDistance(self, individ):
        """Calculul distantei rutelor"""
        #print("individ", individ)
        distances = self.distance[individ[:-1], individ[1:]]
        distance = distances.sum() + self.distance[individ[-1], individ[0]]
        return distance

    def getIndividNumberCities(self, individ):
        return np.unique(individ[:-1], return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def getBestRoute(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestDistance(self, population):
        # calculeaza distanta
        distances = self.getDistances(population)
        return distances.min()

    def getBestNumberCities(self, population):
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        return number_city.max()

    def getKBest(self, fitness_values, size_best):
        # returneaza pozitiile cu cele mai mari valori
        arg_ = np.argpartition(fitness_values,-size_best)
        return arg_[-size_best:]

    def getKWeaks(self, fitness_values, size_best):
        # returneaza pozitiile cu cele mai mari valori
        arg_ = np.argpartition(fitness_values,size_best)
        return arg_[:size_best]

    def getKExtreme(self, fitness_values):
        # returneaza pozitiile cu cele mai mari valori
        # select size of normal population
        #k_normal = TSP.K_BEST+TSP.K_WRONG
        # select best individs
        #k = fitness_values.shape[0]//2-k_normal//2
        arg_ = np.argpartition(fitness_values,-TSP.K_BEST)
        #arg_normal = arg_[-k:k_normal-k]
        arg_best   = arg_[-TSP.K_BEST:]
        # select left individs
        #arg_wrong = np.argpartition(fitness_values, TSP.K_WRONG)[:TSP.K_WRONG]
        #return np.concatenate((arg_best, arg_normal, arg_wrong), axis=0)
        return arg_best

    def getDistances(self, population):
        """calcularea distantei pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.getIndividDistance,
                                        axis=1,
                                        arr=population)

    def getNumberCities(self, population):
        # calculeaza numarul de orase unice
        return np.apply_along_axis(self.getIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def fitness(self, population):
        """ valoarea la fitness este inversul distantei dintre orase si un numar cat mai mare de orase vizitate
        - se calculeaza distanta maxima pentru fiecare individ
        - se calculeaza numarul de orase unice
        """
        # calculeaza distanta
        distances = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # normalizeaza intervalul 0...1
        number_city, mask_cities = self.cityNorm(number_city)
        distances   = self.distanceNorm(distances, mask_cities)
        #print("distances: min {:.3f}, max {:.3f}, mean {:.3f}, std {:.3f}, quatile_25 {:.3f}, quatile_50 {:.3f}, quatile_75 {:.3f}".format(distances.min(), distances.max(), np.mean(distances), np.std(distances),
        #                                                          np.quantile(distances, 0.25), np.quantile(distances, 0.5), np.quantile(distances, 0.75)))
        #print("number_city", number_city)
        #print("neighbor_city", neighbor_city)
        distances = distances**2
        fitness_values = 2*distances*number_city/(distances+number_city)
        return fitness_values

    def childFitness(self, population, city_tolerance):
        """ valoarea la fitness este inversul distantei dintre orase si un numar cat mai mare de orase vizitate
        - se calculeaza distanta maxima pentru fiecare individ
        - se calculeaza numarul de orase unice
        """
        # calculeaza distanta
        distances = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # normalizeaza intervalul 0...1
        number_city = self.cityNormChilds(number_city, city_tolerance)
        distances   = self.distanceNormChilds(distances)
        #print("distances: min {:.3f}, max {:.3f}, mean {:.3f}, std {:.3f}, quatile_25 {:.3f}, quatile_50 {:.3f}, quatile_75 {:.3f}".format(distances.min(), distances.max(), np.mean(distances), np.std(distances),
        #                                                          np.quantile(distances, 0.25), np.quantile(distances, 0.5), np.quantile(distances, 0.75)))
        #print("number_city", number_city)
        #print("neighbor_city", neighbor_city)
        distances = distances**2
        fitness_values = 2*distances*number_city/(distances+number_city)
        return fitness_values

    def distanceNorm(self, distances, mask_cities):
        #max_distance = distances.max()
        prev_min_dist = self.__min_distance
        #print("mask_cities", mask_cities, distances[mask_cities])
        mask_dist = distances[mask_cities]
        if (mask_dist.shape[0] > 0):
            tmp_min = mask_dist.min()
        else:
            tmp_min = prev_min_dist
        #self.__min_distance = prev_min_dist*0.2 + tmp_min*0.8
        self.__min_distance = tmp_min
        #self.__min_distance = distances.min()
        #return (prev_min_dist + self.__min_distance)/(distances + self.__min_distance)
        return (2*self.__min_distance)/(distances + self.__min_distance)

    def cityNorm(self, number_city):
        mask_cities = (number_city==TSP.GENOME_LENGTH)
        return mask_cities.astype(np.float32), mask_cities

    def distanceNormChilds(self, distances):
        return (2*self.__min_distance)/(distances + self.__min_distance)

    def cityNormChilds(self, number_city, city_tolerance):
        mask_cities = (number_city>(TSP.GENOME_LENGTH-city_tolerance))
        return mask_cities.astype(np.float32)*number_city/TSP.GENOME_LENGTH
