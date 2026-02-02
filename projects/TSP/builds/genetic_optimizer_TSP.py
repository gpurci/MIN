#!/usr/bin/python

import numpy as np

class TSP(object):
    """

    """
    # constante pentru setarea algoritmului
    GENERATIONS     = 500 # numarul de generatii
    POPULATION_SIZE = 100 # numarul populatiei
    GENOME_LENGTH   = 4 # numarul de alele
    MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
    CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
    SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
    ELITE_SIZE      = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor

    def __init__(self, name=""):
        self.__name = name
        self.__distance_evolution = np.zeros(5, dtype=np.float32)
        self.__prev_best    = 0
        self.__min_distance = 0

    def __str__(self):
        info = """name: {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    ELITE_SIZE      = {}
    GENERATIONS     = {}""".format(self.__name, TSP.POPULATION_SIZE, TSP.GENOME_LENGTH, TSP.MUTATION_RATE, 
                                    TSP.CROSSOVER_RATE, TSP.SELECT_RATE, TSP.ELITE_SIZE, TSP.GENERATIONS)
        return info

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

    def setParameters(self, **kw):
        TSP.POPULATION_SIZE = kw.get("POPULATION_SIZE", TSP.POPULATION_SIZE)
        TSP.MUTATION_RATE   = kw.get("MUTATION_RATE", TSP.MUTATION_RATE)
        TSP.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE", TSP.CROSSOVER_RATE)
        TSP.SELECT_RATE     = kw.get("SELECT_RATE", TSP.SELECT_RATE)
        TSP.GENERATIONS     = kw.get("GENERATIONS", TSP.GENERATIONS)
        TSP.ELITE_SIZE      = kw.get("ELITE_SIZE", TSP.ELITE_SIZE)

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

    def selectValidPopulation(self, args_parents1, fitness_parents1):
        """selectarea pozitiilor valide pentru parinti 2
        args_parents1    - pozitiile indivizilor ce fac parte din parinte 1
        fitness_parents1 - valorile fitnes cuprinse 0...1
        """
        # select valid parents for parents2, from list of valid parents1
        # 1/3 from parents1 is valid as a parents2
        # create mask of valid parents2 from valid parents1
        total_fitness = fitness_parents1.sum()
        if (total_fitness != 0):
            fitness_parents1 = fitness_parents1/total_fitness
        else:
            fitness_parents1 = None
        # selectare aleatorie din parinti 1 care pot fi si ca parinti 2
        args_valid_parents1 = np.random.choice(args_parents1.shape[0], size=args_parents1.shape[0]//3, p=fitness_parents1)
        # exclude valid parents1 from invalid parents selection
        mask = np.ones(args_parents1.shape[0], dtype=bool)
        #print("mask", mask.shape)
        mask[args_valid_parents1] = False
        # do invalid parents1 for valid parents2
        args_invalid_parents1 = args_parents1[mask]
        args_populations = np.ones(TSP.POPULATION_SIZE, dtype=bool)
        #print("args_populations", args_populations.shape)
        args_populations[args_invalid_parents1] = False
        return np.argwhere(args_populations).reshape(-1)

    def selectParent1(self, fitness_values, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_values - valorile fitnes cuprinse 0...1,
        size           - numarul de parinti in calitate de parinti 1
        """
        # selectare aleatorie a metodei de selectie a parintelui
        select_rate = np.random.uniform(low=0, high=1, size=None)
        #
        if (select_rate < TSP.SELECT_RATE): # selectare aleatori a parintilor 1
            total_fitness = fitness_values.sum()
            if (total_fitness != 0):
                prob_fitness = fitness_values / total_fitness
            else:
                prob_fitness = None
            # selectare aleatorie
            args = np.random.choice(TSP.POPULATION_SIZE, size=size, p=prob_fitness)
        else:
            # selectare secventiala
            args = np.arange(size, dtype=np.int32)
        return args

    def selectIndividParent2(self, parents2_fitness, select_cond, pick):
        """selectarea unui parinte aleator din populatie, in calitate de parinte 2
        parents2_fitness - fitnesul pentru parinte 2 normalizat
        select_cond - o valoare de la 0...2 inclusiv
                        0 - selecteaza aleator parinte 2, metoda roata norocului
                        1 - selecteaza aleator parinte 2, selectie aleatorie cu sanse egale
                        2 - selecteaza aleator parinte 2, selectie aleatorie, dupa valoarea fitnesului
        pick        - suma fitnesului asteptata
        """
        if (select_cond == 0): # selectie dupa compatibilitate, roata norocului
            current = 0
            # roata norocului
            for arg, fitness_value in enumerate(parents2_fitness, 0):
                current += fitness_value
                if (current > pick):
                    break
        elif (select_cond == 1): # selectie aleatorie, cu aceeasi sansa de castig
            arg = np.random.choice(parents2_fitness.shape[0], size=None, p=None)
        elif (select_cond == 2):  # selectie aleatorie, probabilitatea alegerii fiind dictata de valoarea fitness, 
            # selecteaza argumentul parintelui 2
            # parents2_fitness - suma trebuie sa fie 1, vector cu valori cuprinse intre 0...1
            arg = np.random.choice(parents2_fitness.shape[0], size=None, p=parents2_fitness)
        return arg

    def selectParents2(self, fitness_parents2, fitness_partener, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_parents2 - valorile fitness cuprinse 0...1, pentru parinte 2
        fitness_partener - valoarea fitnes a parintelui 1 in calitate de partener
        size             - numarul de parinti in calitate de parinti 2
        """
        total_fitness = fitness_parents2.sum()
        if (total_fitness == 0): # daca avem un fitness invalid (valori zero)
            # selecteava un individ aleatoriu cu probabilitati egale
            select_conds     = np.full(size, 1, dtype=np.int32)
            # selectia cu probabilitati egale, nu foloseste 'parents2_fitness', 'picks'
            parents2_fitness = fitness_parents2
            picks            = select_conds
        else:
            # select condition for all parteners
            p = [TSP.SELECT_RATE/2, 1-TSP.SELECT_RATE, TSP.SELECT_RATE/2]
            """avem 3 metode de selectie a parintelui 2,
                    0 - selecteaza aleator parinte 2, metoda roata norocului
                    1 - selecteaza aleator parinte 2 in dependenta de distributia fitnesului
                    2 - selecteaza aleator parinte 2, cu sanse egale
            """
            select_conds = np.random.choice([0, 1, 2], size=size, p=p)
            """conditii de selectare a partenerului
            0 - selectare partener, unde probabilitatea alegerii este valoarea fitnes a individului (cu cat mai mare valoarea ca atat sansele sunt mai mari)
            1 - selectare partener, roata norocului
            2 - selectare partener, unde probabilitatea alegerii este egala intre parteneri
            """
            # normalizeaza valorile fitness
            parents2_fitness = fitness_parents2 / total_fitness
            # calculeaza valoarea asteptata pentru roata norocului
            # calculeaza 'pick' pentru toti partenerii
            picks = np.random.uniform(low=fitness_partener/total_fitness, high=1, size=size)
        # bucla de selectare parinti 2
        parent_args = []
        for pick, select_cond in zip(picks, select_conds):
            arg = self.selectIndividParent2(parents2_fitness, select_cond, pick)
            parent_args.append(arg)
        return np.array(parent_args, dtype=np.int32)
        
    def crossoverIndivid(self, parent1, parent2, arg, childs, coords, crossover_conds):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        arg     - pozitia din vector
        childs  - vector de indivizi, pentru copii
        coords  - vector de coordonate start, end
        crossover_conds - vector pentru metodele de aplicare a incrucisarii
        """
        # creare un copil fara mostenire
        child       = childs[arg]
        # selectarea diapazonului de mostenire
        start, end  = coords[arg]
        cond = crossover_conds[arg]
        #print("coords ", coords[arg], "cond", cond)
        if (start > end):
            start, end = end, start
        if (cond == 0):
            # copierea rutei din primul parinte
            child[start:end] = parent1[start:end]
            # copierea rutei din cel de al doilea parinte
            child[:start] = parent2[:start]
            child[end:]   = parent2[end:]
        elif (cond == 1):
            # modifica doar genele care sunt diferite
            mask = parent1!=parent2
            mask[[0, -1]] = False # pastreaza orasul de start
            args = np.argwhere(mask).reshape(-1)
            if (args.shape[0] > 1):
                args = args.reshape(-1)
                tmp_size = min(end-start, args.shape[0]//2)
                args = np.random.choice(args, size=tmp_size)
                child[:]    = parent1[:]
                child[args] = parent2[args]

    def crossover(self, parent1, parent2, nbr_childs=1):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # creare un copil fara mostenire
        childs = np.zeros((nbr_childs, TSP.GENOME_LENGTH+1), dtype=np.int32)
        # selectarea diapazonului de mostenire
        coords = np.random.randint(low=1, high=TSP.GENOME_LENGTH+1, size=(nbr_childs, 2))
        # medodele de aplicare a incrucisarii
        # cond 0 -> selectare o zona aleatorie de gene
        #      1 -> se face incrucisare doar la genele diferite
        crossover_conds = np.random.choice([0, 1], size=nbr_childs, p=[0.6, 0.4])
        for arg in range(nbr_childs):
            self.crossoverIndivid(parent1, parent2, arg, childs, coords, crossover_conds)
        return childs

    def mutateIndivid(self, individs, parent1, parent2, arg, coords, mutate_conds):
        """Mutatia genetica a individului
            individs - lista de indivizi
            arg      - pozitia din vector
            coords   - vector de coordonate start, end
            mutate_conds - vector pentru metodele de aplicare a mutatiei
        """
        # selectarea genomurile care vor fi mutate cu locul
        individ = individs[arg]
        cond    = mutate_conds[arg]
        loc1, loc2 = coords[arg]
        if   (cond == 0):
            pass
        elif (cond == 1):
            individ[loc1], individ[loc2] = individ[loc2], individ[loc1]
        elif (cond == 2):
            # modifica doar genele, unde codul genetic al parintilor este identic
            mask = parent1==parent2
            mask[[0, -1]] = False # pastreaza orasul de start
            args_similar = np.argwhere(mask).reshape(-1)
            if (args_similar.shape[0] > 1):
                args_similar = args_similar.reshape(-1)
                # obtine genele similare
                similar_genes = parent1[args_similar]
                # sterge genele care au fost gasite
                mask_valid = np.ones(TSP.GENOME_LENGTH, dtype=bool)
                mask_valid[similar_genes] = False
                # adauga alte gene
                new_gene = np.argwhere(mask_valid).reshape(-1)
                new_gene = np.random.choice(new_gene,     size=2)
                args     = np.random.choice(args_similar, size=2)
                individ[args] = new_gene

    def mutate(self, individs, parent1, parent2):
        """Mutatia genetica a indivizilor, operatie in_place
            individs - lista de indivizi
        """
        # selectarea genomurilor care vor fi mutate cu locul
        # aplicarea operatiei de mutatie pentru tot numarul de indivizi
        nbr_individs = individs.shape[0]
        # prababilitatea pentru fiecare metoda de mutatie
        p = [1-TSP.MUTATION_RATE, TSP.MUTATION_RATE/2, TSP.MUTATION_RATE/2]
        # cond 0 -> nu se aplica operatia de mutatie
        # cond 1 -> se aplica operatia de mutatie, metoda swap
        # cond 2 -> se aplica operatia de mutatie, este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
        mutate_conds = np.random.choice([0, 1, 2], size=nbr_individs, p=p)
        # coordonatele lalele-lor
        coords = np.random.randint(low=1, high=TSP.GENOME_LENGTH+1, size=(nbr_individs, 2))
        # loop pentru fiecare individ
        for arg in range(nbr_individs):
            self.mutateIndivid(individs, parent1, parent2, arg, coords, mutate_conds)
        return individs

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
