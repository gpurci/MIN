#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class SelectParent(RootGA):
    """
    Clasa 'SelectParent', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, method, **kw):
        super().__init__()
        self.__configs  = kw
        self.parent_arg = 0
        self.__setMethods(method)

    def __str__(self):
        info = """SelectParent: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __call__(self):
        return self.__fn(**self.__configs)

    def __unpackMethod(self, method, extern_fn):
        fn = self.selectParentAbstract
        if (method is not None):
            if   (method == "choice"):
                fn = self.selectParentChoice
            elif (method == "rand"):
                fn = self.selectParentRand
            elif (method == "wheel"):
                fn = self.selectParentWheel
            elif (method == "tour"):
                fn = self.selectParentTour
            elif (method == "tour_choice"):
                fn = self.selectParentTourChoice
            elif (method == "rise"):
                fn = self.selectParentRise
            elif (method == "mixt"):
                fn = self.selectParentMixt
            elif ((method == "extern") and (extern_fn is not None)):
                fn = extern_fn
                
        return fn

    def help(self):
        info = """SelectParent:
    metoda: 'choice';      config: None;
    metoda: 'rand';        config: None;
    metoda: 'wheel';       config: None;
    metoda: 'tour';        config: -> "size_subset":7;
    metoda: 'tour_choice'; config: -> "size_subset":7;
    metoda: 'rise';        config: None;
    metoda: 'mixt';        config: -> "p_select":[1/4, 1/4, 1/4, 1/4], "size_subset":7;
    metoda: 'extern';      config: 'extern_kw';\n"""
        print(info)

    def __setMethods(self, method):
        self.__method = method
        self.__extern_fn = self.__configs.pop("extern_fn", None)
        self.__fn = self.__unpackMethod(method, self.__extern_fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if (self.__extern_fn is not None):
            self.__extern_fn.setParameters(**kw)

    def startEpoch(self, fitness_values):
        total_fitness = fitness_values.sum()
        if (total_fitness != 0):
            # calcularea numarului de indivizi valizi, pentru selectie
            size = self.POPULATION_SIZE - int(self.SELECT_RATE*self.POPULATION_SIZE)
            # selectarea celor mai slabi indivizi
            args_weaks = np.argpartition(fitness_values, size)[:size]
            # scoterea indivizilor slabi din cursa pentru parinte
            fitness_values[args_weaks] = 0.
            total_fitness = fitness_values.sum()
            # update: adauga doar cei mai puternici indivizi
            self.fitness_values = fitness_values / total_fitness
        else:
            self.fitness_values = np.full(fitness_values.shape[0], 1./self.POPULATION_SIZE, dtype=np.float32)

    def selectParentAbstract(self, **kw):
        raise NameError("Lipseste metoda '{}' pentru functia de 'SelectionParent': config '{}'".format(self.__method, self.__config))

    def selectParentChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.choice(self.POPULATION_SIZE, size=None, p=self.fitness_values)
        return arg

    def selectParentRand(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.randint(low=0, high=self.POPULATION_SIZE, size=None)
        return arg

    def selectParentWheel(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, roata norocului
        current = 0
        # suma fitnesului asteptata
        pick    = np.random.uniform(low=0, high=1, size=None)
        # roata norocului
        for arg, fitness_value in enumerate(self.fitness_values, 0):
            current += fitness_value
            if (current > pick):
                break
        return arg

    def selectParentTour(self, size_subset=7):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.randint(low=0, high=self.POPULATION_SIZE, size=size_subset)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]

    def selectParentTourChoice(self, size_subset=7):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.choice(self.POPULATION_SIZE, size=size_subset, p=self.fitness_values)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]

    def selectParentRise(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, in crestere
        arg = self.parent_arg
        self.parent_arg += 1
        self.parent_arg = self.parent_arg % self.POPULATION_SIZE
        return arg

    def selectParentMixt(self, size_subset=7, p_select=None):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, mixt
        cond = np.random.choice([0, 1, 2, 3], size=None, p=p_select)
        if   (cond == 0):
            arg = self.selectParentChoice(size_subset)
        elif (cond == 1):
            arg = self.selectParentWheel(size_subset)
        elif (cond == 2):
            arg = self.selectParentTour(size_subset)
        elif (cond == 3):
            arg = self.selectParentRise(size_subset)

        return arg
