#!/usr/bin/python

import numpy as np

class Selection(RootGA):
    """
    Clasa 'Selection', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 si 2
    Functia 'selectParent' are 2 parametri, fitness_value, fitness_partener.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.__config = config

    def __call__(self):
        fn = self.selectionAbstract
        if (self.__config == ""):
            fn = self.mutate
        return fn

    def selectionAbstract(self, parent1, parent2, offspring):
        raise NameError("Configuratie gresita pentru functia de 'Selection'")

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
