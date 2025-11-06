#!/usr/bin/python

import numpy as np

class GeneticAlgorithm(object):
    """

    """

    def __init__(self, name=""):
        self.__name = name
        # constante pentru setarea algoritmului
        self.GENERATIONS     = 500 # numarul de generatii
        self.POPULATION_SIZE = 100 # numarul populatiei
        self.GENOME_LENGTH   = 4 # numarul de alele
        self.MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
        self.CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
        self.SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
        self.ELITE_SIZE      = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor

    def __str__(self):
        info = """name: {}
    GENERATIONS     = {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    ELITE_SIZE      = {}""".format(self.__name, self.GENERATIONS, self.POPULATION_SIZE, self.GENOME_LENGTH, self.MUTATION_RATE, 
                                    self.CROSSOVER_RATE, self.SELECT_RATE, self.ELITE_SIZE)
        return info

    def __call__(self):
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
        raise NameError("Nu este implementat corpul agoritmului genetic")

    def calculateParameterValue(self):
        """Calculeaza:
        - numarul de parinti 1,
        - numarul de parteneri ce are parinte 1
        - numarul de copii care ii poate avea parinte 1 cu parinte 2
        - numarul de copii final ce ii poate avea parinte 1
        """
        # calculeaza populatia din noua generatie, inclusiv elita
        population_size_out_extreme = self.POPULATION_SIZE-self.ELITE_SIZE
        # numarul de parinti 1 care vor fi selectati
        size_parent1 = int(population_size_out_extreme/2)
        # calculeaza numarul de parteneri pentru fiecare parinte 1,
        nbr_parteners = np.zeros(size_parent1, dtype=np.int32)+7
        nbr_parteners[-self.ELITE_SIZE:] = 150
        # calculeaza numarul de copii care poate sa ii aiba parinte 1
        nbr_childrens = np.zeros(size_parent1, dtype=np.int32)+3
        nbr_childrens[-self.ELITE_SIZE:] = 2
        # din numarul total de copii care poate sa ii aiba parinte 1, selecteaza cei mai buni
        nbr_best_childrens = np.ones(size_parent1, dtype=np.int32) # parintii care fac parte din elita pot avea un singur copil
        # calculeaza numarul de copii care ii poate avea, un individ simplu
        k_best_childrens   = (population_size_out_extreme-self.ELITE_SIZE)//(size_parent1-self.ELITE_SIZE)
        # salveaza numarul de copii
        nbr_best_childrens[:-self.ELITE_SIZE] = k_best_childrens
        # calculeaza cati copii, trebuie de adaugat pentru a avea o populatie deplina
        # 
        tmp_big = population_size_out_extreme-(size_parent1-self.ELITE_SIZE)*k_best_childrens-self.ELITE_SIZE
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
        raise NameError("Nu este implementata 'evolutionMonitor'")

    def setElites(self, population, elites):
        raise NameError("Nu este implementata 'setElites'")

    def setParameters(self, **kw):
        self.POPULATION_SIZE = kw.get("POPULATION_SIZE", self.POPULATION_SIZE)
        self.MUTATION_RATE   = kw.get("MUTATION_RATE", self.MUTATION_RATE)
        self.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE", self.CROSSOVER_RATE)
        self.SELECT_RATE     = kw.get("SELECT_RATE", self.SELECT_RATE)
        self.GENERATIONS     = kw.get("GENERATIONS", self.GENERATIONS)
        self.ELITE_SIZE      = kw.get("ELITE_SIZE", self.ELITE_SIZE)

    def clcMetrics(self, population, fitness_values):
        """
        Calculare metrici:
            population - populatia compusa dintr-o lista de indivizi
            fitness_values - valorile fitnes pentru fiecare individ
        """
        raise NameError("Nu este implementata calcularea metricilor")

    def showMetrics(self, generation, d_info):
        """Afisare metrici"""
        metric_info ="Name:{}, Generatia: {}, ".format(self.__name, generation)
        for key in d_info.keys():
            val = d_info[key]
            if (isinstance(val, float)):
                val = round(val, 3)
            metric_info + ="{}: {},".format(key, val)
        print(metric_info)

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        raise NameError("Nu este implementata 'evolutionMonitor'")

    def individReconstruction(self, individ):# TO DO: aplica shift sau permutare pe secvente mai mici
        """Initializare individ, cu drumuri aleatorii si oras de start
        start_gene - orasul de start
        """
        raise NameError("Nu este implementata 'evolutionMonitor'")

    def selectValidPopulation(self, args_parents1):
        """selectarea pozitiilor valide pentru parinti 2
        args_parents1 - pozitiile indivizilor ce fac parte din parinte 1
        """
        args_populations = np.ones(self.POPULATION_SIZE, dtype=bool)
        args_populations[args_parents1] = False
        return np.argwhere(args_populations).reshape(-1)

    def selectParent1(self, fitness_values, args_elites, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_values - valorile fitnes cuprinse 0...1,
        size           - numarul de parinti in calitate de parinti 1
        """
        # selectare aleatorie a metodei de selectie a parintelui
        select_rate = np.random.uniform(low=0, high=1, size=None)
        size -= self.ELITE_SIZE
        #
        if (select_rate < self.SELECT_RATE): # selectare aleatori a parintilor 1
            total_fitness = fitness_values.sum()
            if (total_fitness != 0):
                prob_fitness = fitness_values / total_fitness
            else:
                prob_fitness = None
            # selectare aleatorie
            args = np.random.choice(self.POPULATION_SIZE, size=size, p=prob_fitness)
        else:
            # selectare secventiala
            args = np.arange(size, dtype=np.int32)
        args = np.concatenate((args, args_elites), axis=0)
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

    def selectParents2(self, fitness_parents2:dict, fitness_partener:dict, size):
        """selectarea unui numar 'size' de parinti aleator din populatie, bazandune pe distributia fitness
        fitness_parents2 -  'global_fitness' - valorile fitness cuprinse 0...1, pentru parinte 2
                            'group_fitness'  - valorile fitness cuprinse 0...1, pentru parinte 2
        fitness_partener -  'global_fitness' - valorile fitness cuprinse 0...1, pentru parinte 1
                            'group_fitness'  - valorile fitness cuprinse 0...1, pentru parinte 1
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
            p = [self.SELECT_RATE/2, 1-self.SELECT_RATE, self.SELECT_RATE/2]
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

    def crossoverIndivid(self, parent1, parent2, arg, childs):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        arg     - pozitia din vector
        childs  - vector de indivizi, pentru copii
        """
        # creare un copil fara mostenire
        child    = childs[arg]
        # mosteneste parinte1
        child[:] = parent1[:]
        # modifica doar genele care sunt diferite
        mask = parent1!=parent2
        mask[[0, -1]] = False # pastreaza orasul de start
        args = np.argwhere(mask)
        size = args.shape[0]
        if (size > 4):
            tmp_size = np.random.randint(low=size//3, high=size//2, size=None)
            args = args.reshape(-1)
            args = np.random.permutation(args)[:tmp_size]
            child[args] = parent2[args]
        else:
            start, end = np.random.randint(low=1, high=self.GENOME_LENGTH, size=2)
            if (start > end): start, end = end, start
            # copierea rutei din parintele 2
            child[start:end] = parent2[start:end]

    def crossover(self, parent1, parent2, nbr_childs=1):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # creare un copil fara mostenire
        childs = np.zeros((nbr_childs, self.GENOME_LENGTH+1), dtype=np.int32)
        for arg in range(nbr_childs):
            self.crossoverIndivid(parent1, parent2, arg, childs)
        return childs

    def mutateIndivid(self, parent1, parent2, individs, arg, mutate_conds):
        """Mutatia genetica a individului
            parent1  - parinte 1 pentru individ
            parent2  - parinte 2 pentru individ
            individs - lista de indivizi
            arg      - pozitia din vector
            mutate_conds - vector pentru metodele de aplicare a mutatiei
        """
        # selectarea genomurile care vor fi mutate cu locul
        individ = individs[arg]
        cond    = mutate_conds[arg]
        if   (cond == 0):
            pass
        elif (cond == 1):
            # metoda cel mai apropiat vecin
            # 1 obtine alela (loc)
            # 2 gena conditionala, care cauta cel mai apropiat vecin este alela precedenta (loc-1)
            # 3 seteaza gena de pe (loc)
            # 4 incrementeaza loc, repeta punctul (2, 3) break

            # loc - alela unde va fi aplicata mutatia, cuprinsa intre 1...GENOME_LENGTH-1
            loc = np.random.randint(low=1, high=self.GENOME_LENGTH, size=None)
            # cond_gene - gena dupa care se va cauta cel mai apropiat vecin
            cond_genes     = individ[[loc-1, loc+1]]
            neighbors_gene = self.getNeighbors(cond_genes)
            new_gene       = np.random.permutation(neighbors_gene)[0]
            gene           = individ[loc]
            loc_new        = individ == new_gene
            individ[loc]   = new_gene
            individ[loc_new] = gene
        elif (cond == 2):
            # modifica doar genele, unde codul genetic al parintilor este identic
            mask = parent1==parent2
            mask[[0, -1]] = False # pastreaza orasul de start
            args_similar  = np.argwhere(mask).reshape(-1)
            if (args_similar.shape[0] > 1):
                args_similar = args_similar.reshape(-1)
                # obtine genele similare
                similar_genes = parent1[args_similar]
                # sterge genele care au fost gasite
                mask_valid = np.ones(self.GENOME_LENGTH, dtype=bool)
                mask_valid[similar_genes] = False
                # adauga alte gene
                new_gene = np.argwhere(mask_valid).reshape(-1)
                new_gene = np.random.permutation(new_gene)[:2]
                args     = np.random.permutation(args_similar)[:2]
                individ[args] = new_gene

    def mutate(self, individs, parent1, parent2):
        """Mutatia genetica a indivizilor, operatie in_place
            individs - lista de indivizi
        """
        # selectarea genomurilor care vor fi mutate cu locul
        # aplicarea operatiei de mutatie pentru tot numarul de indivizi
        nbr_individs = individs.shape[0]
        # prababilitatea pentru fiecare metoda de mutatie
        p = [1-self.MUTATION_RATE, self.MUTATION_RATE/2, self.MUTATION_RATE/2]
        # cond 0 -> nu se aplica operatia de mutatie
        # cond 1 -> se aplica operatia de mutatie, metoda vecinul apropiat
        # cond 2 -> se aplica operatia de mutatie, este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
        mutate_conds = np.random.choice([0, 1, 2], size=nbr_individs, p=p)
        # loop pentru fiecare individ
        for arg in range(nbr_individs):
            self.mutateIndivid(parent1, parent2, individs, arg, mutate_conds)
        return individs

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
        similar_args_flag = np.zeros(self.POPULATION_SIZE, dtype=bool)
        # setare toleranta, numarul total de gene
        tolerance = self.GENOME_LENGTH
        # 
        for i in range(self.POPULATION_SIZE-1, -1, -1):
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
        raise NameError("Nu este implementata functia 'permuteSimilarIndivids'!!!")

    def stres(self, population, fitness_values, best_individ, best_distance):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population    - populatia
        best_individ  - individul cu cel mai bun fitness
        best_distance - cea mai buna distanta
        """
        raise NameError("Nu este implementata functia 'permuteSimilarIndivids'!!!")

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

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
        args = np.argpartition(fitness_values,-self.ELITE_SIZE)
        args = args[-self.ELITE_SIZE:]
        return args
