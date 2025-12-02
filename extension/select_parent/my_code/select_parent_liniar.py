#!/usr/bin/python

import numpy as np
from extension.select_parent.my_code.select_parent_base import *
from extension.utils.my_code.softmax import *

class SelectParentLiniar(SelectParentBase):
    """
    Clasa 'SelectParentLiniar', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="SelectParentLiniar", **configs)
        self.__fn = self._unpackMethod(method, 
                                        choice=self.selectParentChoice, 
                                        uniform=self.selectParentUniform, 
                                        wheel=self.selectParentWheel,  
                                        sort_wheel=self.selectParentSortWheel, 
                                        tour=self.selectParentTour, 
                                        tour_choice=self.selectParentTourChoice, 
                                        mixt=self.selectParentMixt, 
                                    )

    def __call__(self):
        return self.__fn(**self._configs)

    def help(self):
        info = """SelectParentLiniar:
    metoda: 'choice';      config: None;
    metoda: 'uniform';     config: None;
    metoda: 'wheel';       config: None;
    metoda: 'sort_wheel';  config: None;
    metoda: 'tour';        config: -> "size_subset":7;
    metoda: 'tour_choice'; config: -> "size_subset":7;
    metoda: 'mixt';        config: -> "p_select":[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], "size_subset":7;\n"""
        print(info)

    def startEpoch(self, fitness_values):
        fitness_values = np.nan_to_num(fitness_values, nan=1e-9)
        min_fitness = fitness_values.min()
        # calcularea numarului de indivizi valizi, pentru selectie
        size = self.POPULATION_SIZE - int(self.SELECT_RATE*self.POPULATION_SIZE)
        # selectarea celor mai slabi indivizi
        args_weaks = np.argpartition(fitness_values, size)[:size]
        # scoterea indivizilor slabi din cursa pentru parinte
        fitness_values[args_weaks] = min_fitness
        # update: adauga doar cei mai puternici indivizi
        self.setFitnessValues(softmax(fitness_values))

    def selectParentMixt(self, size_subset=7, p_select=None):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, mixt
        cond = np.random.choice([0, 1, 2, 3, 4, 5], size=None, p=p_select)
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

        return arg
