#!/usr/bin/python

import numpy as np
from root_GA import *

class SelectParent(RootGA):
    """
    Clasa 'SelectParent', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__()
        self.setConfig(config)
        self.parent_arg = 0

    def __call__(self):
        return self.fn()

    def __config_fn(self):
        self.fn = self.selectParentAbstract
        if (self.__config is not None):
            if   (self.__config == "choice"):
                self.fn = self.selectParentChoice
            elif (self.__config == "roata"):
                self.fn = self.selectParentWheel
            elif (self.__config == "turneu"):
                self.fn = self.selectParentTour
            elif (self.__config == "turneu_choice"):
                self.fn = self.selectParentTourChoice
            elif (self.__config == "crescator"):
                self.fn = self.selectParentRise
            elif (self.__config == "mixt"):
                self.fn = self.selectParentMixt
                
        else:
            pass

    def help(self):
        info = """SelectParent: 
        metode de config: 'choice', 'roata', 'turneu', 'turneu_choice', 'crescator', 'mixt'\n"""
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def startEpoch(self, fitness_values):
        total_fitness = fitness_values.sum()
        if (total_fitness != 0):
            # calcularea numarului de indivizi valizi, pentru selectie
            size = int(self.SELECT_RATE*self.POPULATION_SIZE)
            # selectarea celor mai slabi indivizi
            args_weaks = np.argpartition(fitness_values, size)[:size]
            # scoterea indivizilor slabi din cursa pentru parinte
            fitness_values[args_weaks] = 0.
            total_fitness = fitness_values.sum()
            # update: adauga doar cei mai puternici indivizi
            self.fitness_values = fitness_values / total_fitness
        else:
            self.fitness_values = np.full(fitness_values.shape[0], 1./self.POPULATION_SIZE, dtype=np.float32)

    def selectParentAbstract(self):
        raise NameError("Lipseste configuratia pentru functia de 'SelectionParent': config '{}'".format(self.__config))

    def selectParentChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.choice(self.POPULATION_SIZE, size=None, p=self.fitness_values)
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

    def selectParentTour(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.randint(low=0, high=self.POPULATION_SIZE, size=7)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]

    def selectParentTourChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.choice(self.POPULATION_SIZE, size=7, p=self.fitness_values)
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

    def selectParentMixt(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, mixt
        cond = np.random.choice([0, 1, 2, 3], size=None, p=None)
        if   (cond == 0):
            arg = self.selectParentChoice()
        elif (cond == 1):
            arg = self.selectParentWheel()
        elif (cond == 2):
            arg = self.selectParentTour()
        elif (cond == 3):
            arg = self.selectParentRise()

        return arg
