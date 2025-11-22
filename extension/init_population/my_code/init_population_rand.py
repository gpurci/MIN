#!/usr/bin/python

import numpy as np
from extension.init_population.my_code.init_population_base import *

class InitPopulationRand(InitPopulationBase):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self):
        super().__init__(method, name="RecInitPopulation", **configs)
        self.__fn = self._unpackMethod(method, 
                                        TSP_rand=self.initPopulationsTSPRand, 
                                        TTP_rand=self.initPopulationsTTPRand,
                                    )

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self._configs)

    def help(self):
        info = """InitPopulationRand:
    metoda: 'TTP_rand';  config: None;
    metoda: 'TSP_rand';  config: None;\n"""
        print(info)

    # initPopulationsTSPRand -------------------------------------
    def initPopulationsTSPRand(self, population_size=-1, genoms=None):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga individ in genome
            genoms.add(tsp=np.random.permutation(individ))
        # adauga indivizi in noua generatie
        genoms.saveInit()
        print("population {}".format(genoms.shape))
    # initPopulationsTSPRand =====================================

    # initPopulationsTTPRand -------------------------------------
    def initPopulationsTTPRand(self, population_size=-1, genoms=None):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga tsp_individ in genome
            kp_individ = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            genoms.add(tsp=np.random.permutation(tsp_individ), kp=kp_individ)
        # adauga indivizi in noua generatie
        genoms.saveInit()
        print("population {}".format(genoms.shape))
    # initPopulationsTTPRand =====================================
