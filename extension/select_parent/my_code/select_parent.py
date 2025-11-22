#!/usr/bin/python

import numpy as np
from extension.select_parent.my_code.select_parent_base import *

class SelectParent(SelectParentBase):
    """
    Clasa 'SelectParent', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="SelectParent", **configs)
        self.__fn = self._unpackMethod(method, 
                                        choice=self.selectParentChoice, 
                                        uniform=self.selectParentUniform, 
                                        wheel=self.selectParentWheel,  
                                        sort_wheel=self.selectParentSortWheel, 
                                        tour=self.selectParentTour, 
                                        tour_choice=self.selectParentTourChoice, 
                                        rise=self.selectParentRise, 
                                        mixt=self.selectParentMixt, 
                                    )
        self.parent_arg = 0

    def __call__(self):
        return self.__fn(**self._configs)

    def help(self):
        info = """SelectParent:
    metoda: 'choice';      config: None;
    metoda: 'uniform';     config: None;
    metoda: 'wheel';       config: None;
    metoda: 'sort_wheel';  config: None;
    metoda: 'tour';        config: -> "size_subset":7;
    metoda: 'tour_choice'; config: -> "size_subset":7;
    metoda: 'rise';        config: None;
    metoda: 'mixt';        config: -> "p_select":[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7], "size_subset":7;\n"""
        print(info)

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

    def selectParentChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.choice(self.POPULATION_SIZE, size=None, p=self.fitness_values)
        return arg

    def selectParentUniform(self):
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

    def selectParentSortWheel(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, roata norocului
        current = 0
        # suma fitnesului asteptata
        pick    = np.random.uniform(low=0, high=1, size=None)
        # roata norocului
        sort_args = np.argsort(self.fitness_values)
        for arg, fitness_value in enumerate(self.fitness_values[sort_args], 0):
            current += fitness_value
            if (current > pick):
                break
        return sort_args[arg]

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
        cond = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=None, p=p_select)
        if   (cond == 0):
            arg = self.selectParentChoice()
        elif (cond == 1):
            arg = self.selectParentUniform()
        elif (cond == 2):
            arg = self.selectParentWheel()
        elif (cond == 3):
            arg = self.selectParentSortWheel()
        elif (cond == 4):
            arg = self.selectParentTour(size_subset)
        elif (cond == 5):
            arg = self.selectParentTourChoice(size_subset)
        elif (cond == 6):
            arg = self.selectParentRise()

        return arg
