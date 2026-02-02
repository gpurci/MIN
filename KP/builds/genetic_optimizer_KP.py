#!/usr/bin/python
import numpy as np

class KP(object):
    # constante pentru setarea algoritmului
    GENERATIONS     = 500 # numarul de generatii
    POPULATION_SIZE = 100 # numarul populatiei
    GENOME_LENGTH   = 4 # numarul de alele
    MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
    CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
    SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
    ELITE_SIZE      = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor
    W               = 0   # greutatea maxima admisibila

    def __init__(self, name=""):
        """
        ??????????????
        
        """
        self.__name = name
        self.__profit_evolution = np.zeros(5, dtype=np.float32)
        self.__weight_evolution = np.zeros(5, dtype=np.float32)

    def __str__(self):
        info = """name: {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    ELITE_SIZE      = {}
    GENERATIONS     = {}""".format(self.__name, KP.POPULATION_SIZE, KP.GENOME_LENGTH, KP.MUTATION_RATE, 
                                    KP.CROSSOVER_RATE, KP.SELECT_RATE, KP.ELITE_SIZE, KP.GENERATIONS)
        return info

    def __call__(self, items_section:"pd.Dataframe", population=None):
        """
        Populatia este compusa din indivizi ce au un fitnes mai slab si elita care are cel mai mare fitness.
        Indivizii sunt compusi din alele, (o alela este un numar binar 0..1)
        Numarul populatiei totale este 'POPULATION_SIZE', numarul elitei este 'ELITE_SIZE'.
        Indivizii care alcatuiesc elita sunt pusi in coada populatiei, pentru a face posibil ca unii indivizi din elita sa se incruciseze si cu indivizi din populatia obisnuita.
        Indivizii care fac parte din elita pot avea un numar mai mare de parteneri, dar un numar mic de copii pentru a evita cazuri de minim local.
        Indivizii din populatia simpla au numar mai mic de parteneri dar un numar mai mare de copii, pentru a diversifica populatia.
        """
        # 
        self.__runChecks(items_section)
        # initiaizarea populatiei
        if (population is None):
            population = self.initPopulation(KP.POPULATION_SIZE)
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
        for generation in range(KP.GENERATIONS):
            # nasterea unei noi generatii
            new_population = []
            # selectarea pozitiilor pentru parintii 1
            args_parents1      = self.selectParent1(fitness_values, args_elite, size_parent1)
            #print("args_parents1", np.unique(args_parents1, return_counts=True)[1])
            # selectarea parintilor 2, care sunt diferiti de parinti 1
            valid_arg_parents2 = self.selectValidPopulation(args_parents1, fitness_values[args_parents1])
            valid_population_parents2 = population[valid_arg_parents2]
            #print("valid_population_parents2 {}".format(valid_population_parents2.shape))
            valid_fitness_parents2    = fitness_values[valid_arg_parents2]
            # bucla completarea noii generatii cu cei mai buni copii 
            for arg_parent1, nbr_partener, nbr_children, nbr_best_childs in zip(args_parents1, nbr_parteners, nbr_childrens, nbr_best_childrens):
                p1_childrens  = []
                # selectarea celui de al doilea parinte
                args_parents2 = self.selectParent2(valid_fitness_parents2, fitness_values[arg_parent1], nbr_partener)
                parent1       = population[arg_parent1]
                # bucla nastere copii, parinte 1 cu toate partenerele
                for arg_parent2 in args_parents2:
                    parents2  = valid_population_parents2[arg_parent2]
                    # incrucisarea parintilor
                    # nasterea copiilor pentru parintele 1 cu toate partenerele sale
                    childrens = self.crossover(parent1, parents2, nbr_children)
                    # mutatii
                    childrens = self.mutate(childrens) # in_place operation
                    # adauga noii copii, la numarul total de copii a parinte 1 cu toate partenerele sale
                    p1_childrens.append(childrens)
                # concatenarea tuturor copiilor pentru parintele 1
                p1_childrens = np.concatenate(p1_childrens, axis=0)
                # calcularea fitnesului pentru copii parintelui 1
                childs_fitness_values = self.childFitness(p1_childrens)
                #print("childrens {}, childs_fitness_values {}, shape {}".format(childrens, childs_fitness_values, childrens.shape))
                #print("childrens shape {}, nbr part {}, nbr_child {}, nbr_best_child {}".format(p1_childrens.shape, nbr_partener, nbr_children, nbr_best_childs))
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
            #print("size new population: {}".format(population.shape))
            # calculare fitness
            fitness_values = self.fitness(population)
            # obtinerea pozitiei pentru elite
            args_elite      = self.getArgElite(fitness_values)
            # calculare metrici
            best_individ, best_fitness, best_profit, best_weight, weight, profit = self.clcMetrics(population, fitness_values)
            # monitorizarea evolutiei
            self.evolutionMonitor(best_profit, best_weight)
            #self.log(population, fitness_values, args_elite, elite_individs, best_distance)
            # adaugare stres in populatie atunci cand lipseste progresul
            self.stres(population, best_individ, best_profit, best_weight)
            # afisare metrici
            self.showMetrics(generation, best_individ, best_fitness, best_profit, best_weight, weight, profit)
            # inhiba fitnesul pentru elita pentru a da sansa si altor indivizi sa fie selectati, indepartam minim local
            fitness_values[-KP.ELITE_SIZE:] *= 0.8

        return best_individ, population

    def calculateParameterValue(self):
        """Calculeaza:
        - numarul de parinti 1,
        - numarul de parteneri ce are parinte 1
        - numarul de copii care ii poate avea parinte 1 cu parinte 2
        - numarul de copii final ce ii poate avea parinte 1
        """
        # calculeaza populatia din noua generatie, inclusiv elita
        population_size_out_extreme = KP.POPULATION_SIZE-KP.ELITE_SIZE
        # numarul de parinti 1 care vor fi selectati
        size_parent1 = int(population_size_out_extreme/2.2)
        # calculeaza numarul de parteneri pentru fiecare parinte 1,
        nbr_parteners = np.zeros(size_parent1, dtype=np.int32)+7
        nbr_parteners[-KP.ELITE_SIZE:] = 150
        # calculeaza numarul de copii care poate sa ii aiba parinte 1
        nbr_childrens = np.zeros(size_parent1, dtype=np.int32)+3
        nbr_childrens[-KP.ELITE_SIZE:] = 2
        # din numarul total de copii care poate sa ii aiba parinte 1, selecteaza cei mai buni
        nbr_best_childrens = np.ones(size_parent1, dtype=np.int32) # parintii care fac parte din elita pot avea un singur copil
        # calculeaza numarul de copii care ii poate avea, un individ simplu
        k_best_childrens   = (population_size_out_extreme-KP.ELITE_SIZE)//(size_parent1-KP.ELITE_SIZE)
        # salveaza numarul de copii
        nbr_best_childrens[:-KP.ELITE_SIZE] = k_best_childrens
        # calculeaza cati copii, trebuie de adaugat pentru a avea o populatie deplina
        # 
        tmp_big = population_size_out_extreme-(size_parent1-KP.ELITE_SIZE)*k_best_childrens-KP.ELITE_SIZE
        # adauga pentru primii indivizi parinte 1, cate un copil pentru a avea o populatie deplina
        nbr_best_childrens[:tmp_big]    = k_best_childrens+1
        # afisare parametri 
        #print("nbr_best_childrens", nbr_best_childrens)
        #print("nbr_parteners", nbr_parteners)
        #print("nbr_childrens", nbr_childrens)
        return size_parent1, nbr_parteners, nbr_childrens, nbr_best_childrens

    def evolutionMonitor(self, best_profit: float, best_weight: float):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        best_profit - cel mai bun profit
        best_weight - cea mai buna greutate
        """
        # monitorizare profit
        self.__profit_evolution[:-1] = self.__profit_evolution[1:]
        self.__profit_evolution[-1]  = best_profit
        # monitorizare greutate
        self.__weight_evolution[:-1] = self.__weight_evolution[1:]
        self.__weight_evolution[-1]  = best_weight

    def setExtreme(self, population, extremes):
        if (population is None):
            population = self.initPopulation(KP.POPULATION_SIZE)
        fitness_values = self.fitness(population)
        args = self.getArgWeaks(fitness_values, extremes.shape[0])
        population[args] = extremes
        return population

    def setParameters(self, **kw):
        KP.POPULATION_SIZE = kw.get("POPULATION_SIZE", KP.POPULATION_SIZE)
        KP.MUTATION_RATE   = kw.get("MUTATION_RATE",   KP.MUTATION_RATE)
        KP.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE", KP.CROSSOVER_RATE)
        KP.SELECT_RATE     = kw.get("SELECT_RATE", KP.SELECT_RATE)
        KP.GENERATIONS     = kw.get("GENERATIONS", KP.GENERATIONS)
        KP.ELITE_SIZE      = kw.get("ELITE_SIZE",  KP.ELITE_SIZE)
        KP.W               = kw.get("W", KP.W)

    def log(self, population, fitness_values, args_elite, elite_individs, best_distance):
        if (self.__prev_best < best_distance):
            args_ones = np.argwhere(fitness_values[args_elite]==1).reshape(-1)
            print("extreme fittness {}".format(fitness_values[args_elite]))
            print("test extreme fittness {}".format(self.childFitness(population[args_elite])))
            for arg in args_ones:
                best_individ  = population[args_elite][arg]
                best_distance = self.getIndividProfit(best_individ)
                print("best_distance {}".format(best_distance))
                
            print("prev extreme fittness {}".format(self.childFitness(elite_individs)))
            print("prev extreme distance {}".format(self.getWeights(elite_individs)))

        self.__prev_best = best_distance

    def clcMetrics(self, population, fitness_values):
        """
        Calculare metrici:
            population - populatia compusa dintr-o lista de indivizi
            fitness_values - valorile fitnes pentru fiecare individ
        """
        # obtinerea celui mai bun individ
        arg_best = self.getArgBest(fitness_values)
        # selectarea celui mai bun individ
        best_individ = population[arg_best]
        best_fitness = fitness_values[arg_best]
        best_profit  = self.getIndividProfit(best_individ)
        best_weight  = self.getIndividWeight(best_individ)
        # selectarea celei mai mici greutati din intreaga populatie
        weight = self.getBestWeight(population)
        # selectarea celui mai mare profit din intreaga populatie
        profit = self.getBestProfit(population)

        return best_individ, best_fitness, best_profit, best_weight, weight, profit

    def showMetrics(self, generation, best_individ, best_fitness, best_profit, best_weight, weight, profit):
        """Afisare metrici"""
        metric_info ="""{}, Generatia: {}, Fitness {:.3f}, Profit: {:.3f} Greutate: {:.3f}, Max profit {:.3f}, Min greutate {:.3f}""".format(self.__name,
            generation, best_fitness, best_profit, best_weight, weight, profit)
        print(metric_info)

    def __runChecks(self, items_section: "pd.Dataframe"):
        if (items_section is not None):
            self.items_section = items_section
            # update numarul de orase
            KP.GENOME_LENGTH = self.items_section.shape[0]
        else:
            raise NameError("Parametrul 'items_section' este o valoare 'None'")

    def setDataset(self, items_section):
        if (items_section is not None):
            self.items_section = items_section
            # update numarul de orase
            KP.GENOME_LENGTH = self.items_section.shape[0]
        else:
            raise NameError("Parametrul 'items_section' este o valoare 'None'")

    def initFuncParameters(self, population):
        self.min_weight = self.getBestWeight(population)
        self.max_profit = self.getBestProfit(population)

    def getIndivid(self):
        """Returneaza un individ"""
        return np.random.randint(low=0, high=2, size=KP.GENOME_LENGTH)

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei"""
        if (population_size == -1):
            population_size = KP.POPULATION_SIZE
        # 
        size = (population_size, KP.GENOME_LENGTH)
        population = np.random.randint(low=0, high=2, size=size)
        return population

    def selectValidPopulation(self, args_parents1, fitness_values_parents1):
        """selectarea pozitiilor valide pentru parinti 2
        args_parents1  - pozitiile indivizilor ce fac parte din parinte 1
        fitness_values - valorile fitnes cuprinse 0...1
        """
        # select valid parents for parents2, from list of valid parents1
        # 1/3 from parents1 is valid as a parents2
        # create mask of valid parents2 from valid parents1
        total_fitness = fitness_parents1.sum()
        if (total_fitness != 0):
            fitness_parents1 = fitness_values_parents1/total_fitness
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
        args_populations = np.ones(KP.POPULATION_SIZE, dtype=bool)
        #print("args_populations", args_populations.shape)
        args_populations[args_invalid_parents1] = False
        return np.argwhere(args_populations).reshape(-1)

    def selectParent1(self, fitness_parents1, args_elite, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_parents1 - valorile fitnes cuprinse 0...1, pentru parinte 1
        args_elite       - pozitiile indivizilor ce fac parte din elita
        size             - numarul de parinti in calitate de parinti 1
        """
        # selectare aleatorie a metodei de selectie a parintelui
        select_rate = np.random.uniform(low=0, high=1, size=None)
        size -= KP.ELITE_SIZE # indivizii din elita, default sunt adaugati ca parinte 1
        # 
        if (select_rate < KP.SELECT_RATE): # selectare aleatori a parintilor 1
            prob_fitness = fitness_parents1.copy()
            prob_fitness[args_elite] = 0. # indivizii din elita default sunt adaugati ca parinte 1
            total_fitness = prob_fitness.sum()
            if (total_fitness != 0):
                prob_fitness = prob_fitness / total_fitness
            else:
                prob_fitness = None
            # selectare aleatorie
            args = np.random.choice(KP.POPULATION_SIZE, size=size, p=prob_fitness)
        else:
            # selectare secventiala
            args = np.arange(size, dtype=np.int32)
        # adaugare indivizii din elita, in lista parinti 1
        args = np.concatenate((args, args_elite), axis=0)
        return args

    def selectIndividParent2(self, parents2_fitness, select_cond, pick):
        """selectarea unui parinte aleator din populatie, in calitate de parinte 2
        parents2_fitness - fitnesul pentru parinte 2 normalizat
        select_cond - o valoare de la 0...2 inclusiv
                        0 - selecteaza aleator parinte 2 in dependenta de distributia fitnesului
                        1 - selecteaza aleator parinte 2 in dependenta de suma distributiei fitnesului
                        2 - selecteaza aleator parinte 2, cu sanse egale
        pick        - suma fitnesului asteptata
        """
        if (select_cond == 1): # selectie dupa compatibilitate
            current = 0
            # roata norocului
            for arg, fitness_value in enumerate(parents2_fitness, 0):
                current += fitness_value
                if (current > pick):
                    break
        elif (select_cond == 2): # selectie dupa compatibilitate
            arg = np.random.choice(parents2_fitness.shape[0], size=None, p=None)
        else: # selectie aleatorie
            # selecteaza argumentul parintelui 2
            arg = np.random.choice(parents2_fitness.shape[0], size=None, p=parents2_fitness)
        return arg

    def selectParent2(self, fitness_parents2, fitness_partener, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_parents2 - valorile fitness cuprinse 0...1, pentru parinte 2
        fitness_partener - valoarea fitnes a parintelui 1 in calitate de partener
        size             - numarul de parinti in calitate de parinti 2
        """
        total_fitness    = fitness_parents2.sum()
        if (total_fitness == 0):
            select_conds = np.full(size, 2, dtype=np.int32)
            parents2_fitness = fitness_parents2
            picks        = select_conds
        else:
            # select condition for all parteners
            p = [KP.SELECT_RATE/2, 1-KP.SELECT_RATE, KP.SELECT_RATE/2]
            """avem 3 metode de selectie a parintelui 2,
                    0 - selecteaza aleator parinte 2 in dependenta de distributia fitnesului
                    1 - selecteaza aleator parinte 2 in dependenta de suma distributiei fitnesului, cu cea mai mare sansa de selectie
                    2 - selecteaza aleator parinte 2, cu sanse egale
            """
            select_conds = np.random.choice([0, 1, 2], size=size, p=p)
            parents2_fitness = fitness_parents2 / total_fitness
            total_fitness    = np.sum(parents2_fitness, axis=None)
            # 
            low  = fitness_partener / total_fitness
            high = total_fitness
            # ensure valid range
            if (low > high):
                low, high = high, low
            # calculate pick for all parteners
            picks = np.random.uniform(low=low, high=high, size=size)
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
            # copierea rutei din primul parinte
            args = np.argwhere(parent1!=parent2)
            if (args.shape[0] > 1):
                args = args.reshape(-1)
                args = np.random.choice(args, size=args.shape[0]//2)
                child[:] = parent1[:]
                child[args] = parent2[args]

    def crossover(self, parent1, parent2, nbr_childs=1):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # creare un copil fara mostenire
        childs = np.zeros((nbr_childs, KP.GENOME_LENGTH), dtype=np.int32)
        # selectarea diapazonului de mostenire
        coords = np.random.randint(low=1, high=KP.GENOME_LENGTH, size=(nbr_childs, 2))
        # medodele de aplicare a incrucisarii
        # cond 0 -> selectare o zona aleatorie de gene
        #      1 -> se face incrucisare doar la genele diferite
        crossover_conds = np.random.choice([0, 1], size=nbr_childs, p=[0.7, 0.3])
        for arg in range(nbr_childs):
            self.crossoverIndivid(parent1, parent2, arg, childs, coords, crossover_conds)
        return childs

    def mutateIndivid(self, individs, arg, coords, mutate_conds):
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
        if (cond == 0):
            pass
        elif (cond == 1):
            if (individ[loc1] != individ[loc2]):
                individ[loc1], individ[loc2] = individ[loc2], individ[loc1]
            else:
                individ[loc1] = np.abs(individ[loc1]-1)
        elif (cond == 2):
            if (individ[loc1] != individ[loc2]):
                individ[loc1], individ[loc2] = individ[loc2], individ[loc1]
            else:
                individ[loc2] = np.abs(individ[loc2]-1)
        elif (cond == 3):
            if (individ[loc1] != individ[loc2]):
                individ[loc1], individ[loc2] = individ[loc2], individ[loc1]
            else:
                individ[loc1] = np.abs(individ[loc2]-1)
                individ[loc2] = individ[loc1]

    def mutate(self, individs):
        """Mutatia genetica a indivizilor, operatie in_place
            individs - lista de indivizi
        """
        # selectarea genomurilor care vor fi mutate cu locul
        # aplicarea operatiei de mutatie pentru tot numarul de indivizi
        nbr_individs = individs.shape[0]
        # prababilitatea pentru fiecare metoda de mutatie
        p = [1-KP.MUTATION_RATE, KP.MUTATION_RATE/3, KP.MUTATION_RATE/3, KP.MUTATION_RATE/3]
        # cond 0 -> nu se aplica operatia de mutatie
        # cond 1 -> se aplica operatia de mutatie, metoda swap, caz ca genele sunt egale inverseaza loc1
        # cond 2 -> se aplica operatia de mutatie, metoda swap, caz ca genele sunt egale inverseaza loc2
        # cond 2 -> se aplica operatia de mutatie, metoda swap, caz ca genele sunt egale inverseaza ambele locatii
        mutate_conds = np.random.choice([0, 1, 2, 3], size=nbr_individs, p=p)
        # coordonatele lalele-lor
        coords = np.random.randint(low=1, high=KP.GENOME_LENGTH, size=(nbr_individs, 2))
        # loop pentru fiecare individ
        for arg in range(nbr_individs):
            self.mutateIndivid(individs, arg, coords, mutate_conds)
        return individs

    def genomeGroupsIndivid(self, individ, individ_T):# TO DO
        """
        Cauta secvente identice de cod, in codul genetic al unui individ,
        individ   - vector compus din codul genetic
        individ_T - individ, cu codul genetic (KP.GENOME_LENGTH, 1)
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
        similar_args_flag = np.zeros(KP.POPULATION_SIZE, dtype=bool)
        # setare toleranta, numarul total de gene
        tolerance = KP.GENOME_LENGTH
        # 
        for i in range(KP.POPULATION_SIZE-1, -1, -1):
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
        print("Similar individs: size {}, population size: {}".format(similar_args_flag.sum(), KP.POPULATION_SIZE))
        for i in range(KP.POPULATION_SIZE):
            if (similar_args_flag[i]):
                population[i][:] = self.getIndivid()

    def stres(self, population, best_individ, best_profit, best_weight):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population   - populatia
        best_individ - individul cu cel mai bun fitness
        best_profit  - cel mai bun profit
        best_weight  - cea mai buna greutate
        """
        stres_profit = np.allclose(self.__profit_evolution.mean(), best_profit, rtol=1e-03, atol=1e-08)
        stres_weight = np.allclose(self.__weight_evolution.mean(), best_weight, rtol=1e-03, atol=1e-08)
        if (stres_profit or stres_weight):
            self.__profit_evolution[:] = 0
            self.__weight_evolution[:] = 0
            KP.MUTATION_RATE = 0.9
            self.permuteSimilarIndivids(population)
        else:
            KP.MUTATION_RATE *= 0.9

    def getIndividProfit(self, individ):
        """Calculul profit pentru un singur individ"""
        profit = self.items_section["PROFIT"]
        profit = individ*profit
        return profit.sum()

    def getIndividWeight(self, individ):
        """Calculul greutate pentru un singur individ"""
        weights = self.items_section["WEIGHT"]
        weights = individ*weights
        return weights.sum()

    def getArgBest(self, fitness_values):
        """Calculul cel mai bun individ"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestWeight(self, population):
        """Calculul cea ma buna greutate"""
        weights = self.getWeights(population)
        return np.min(weights[np.nonzero(weights)])

    def getBestProfit(self, population):
        """Calculul cel mai bun profit"""
        profits = self.getProfits(population)
        return profits.max()

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
        args = np.argpartition(fitness_values,-KP.ELITE_SIZE)
        args = args[-KP.ELITE_SIZE:]
        return args

    def getWeights(self, population):
        """Calculeaza greutatea pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.getIndividWeight,
                                        axis=1,
                                        arr=population)

    def getProfits(self, population):
        """Calculeaza profitul pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.getIndividProfit,
                                        axis=1,
                                        arr=population)

    def fitness(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea greutatii este invers normalizata, iar valoarea profitului direct normalizata
        population - populatia, vector de indivizi
        """
        weights = self.getWeights(population)
        # calculeaza numarul de orase unice
        profits = self.getProfits(population)
        # normalizeaza intervalul 0...1
        profits = self.profitNorm(profits)
        weights = self.weightNorm(weights, profits)
        #print("weights", weights.max())
        fitness_values = 2*weights*profits/(weights+profits+1e-7)# adaugam epsilon pentru a inlatura impartirea la 0
        return fitness_values

    def weightNorm(self, weights, profits):
        """
        Normalizam greutatile, intervalul 0...1,
        weights - greutatile pentru toti indivizii din populatie
        cazuri posibile (greutati interzise):
                         weights == 0  - norm 0
                         weights > 'W' - norm 0
                         min din 'weights' == 0, min trebuie de calculat dupa o valoare mai mare decat 0!!!
        """
        # calculam greutatea minima
        mask = (weights!=0)
        mask_weights = weights[mask]
        if (mask_weights.shape[0] > 0):
            min_weight = mask_weights.min()
            if (min_weight > KP.W):
                CAPACITY = mask_weights.mean()
                #args_elite_profits = []
            else:
                CAPACITY = KP.W
                #args_elite_profits = self.getArgElite(profits)

        # verificam valoarea maxima admisibila 'W'
        mask &= (weights > CAPACITY)
        #print("mask > W", mask.sum(), weights.shape, min_weight, CAPACITY)
        # calculam valoarea greutatii normalizate
        norm_weights = weights/CAPACITY
        # punem pe zero greutatile interzise
        norm_weights[mask] = 1e-1
        #if (CAPACITY == KP.W):
        #    norm_weights[args_elite_profits] = 1.
        return norm_weights

    def profitNorm(self, profits):
        """
        Normalizam profitul intervalul 0...1,
        profits - profiturile pentru toti indivizii din populatie
        cazuri posibile :
                         max din 'profits' == 0, adaugam epsilon 1e-7
        """
        max_profit   = profits.max()+1e-7 # adaugam epsilon pentru a inlatura impartirea la 0
        norm_profits = profits/max_profit
        return norm_profits

    def childFitness(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea greutatii este invers normalizata, iar valoarea profitului direct normalizata
        population - populatia, vector de indivizi
        """
        weights = self.getWeights(population)
        # calculeaza numarul de orase unice
        profits = self.getProfits(population)
        # normalizeaza intervalul 0...1
        profits = self.childProfitNorm(profits)
        weights = self.childWeightNorm(weights)
        fitness_values = 2*weights*profits/(weights+profits+1e-7)# adaugam epsilon pentru a inlatura impartirea la 0
        return fitness_values

    def childWeightNorm(self, weights):
        """
        Normalizam greutatile, intervalul 0...1,
        weights - greutatile pentru toti indivizii din populatie
        cazuri posibile (greutati interzise):
                         weights == 0  - norm 0
                         weights > 'W' - norm 0
                         min din 'weights' == 0, min trebuie de calculat dupa o valoare mai mare decat 0!!!
        """
        # calculam greutatea minima
        mask = (weights!=0)
        mask_weights = weights[mask]
        if (mask_weights.shape[0] > 0):
            min_weight = mask_weights.min()
            if (min_weight > KP.W):
                CAPACITY = mask_weights.mean()
            else:
                CAPACITY = KP.W

        # verificam valoarea maxima admisibila 'W'
        mask &= (weights > CAPACITY)
        #print("mask > W", mask.sum(), weights.shape, min_weight, CAPACITY)
        # calculam valoarea greutatii normalizate
        norm_weights = weights/CAPACITY
        # punem pe zero greutatile interzise
        norm_weights[mask] = 1e-1
        return norm_weights

    def childProfitNorm(self, profits):
        """
        Normalizam profitul intervalul 0...1,
        profits - profiturile pentru toti indivizii din populatie
        cazuri posibile :
                         max din 'profits' == 0, adaugam epsilon 1e-7
        """
        max_profit   = profits.max()+1e-7 # adaugam epsilon pentru a inlatura impartirea la 0
        norm_profits = profits/max_profit
        return norm_profits
