#!/usr/bin/python

import numpy as np
from extension.init_population.init_population_base import *

class InitPopulationRand(InitPopulationBase):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="InitPopulationRand", **configs)
        self.__fn = self._unpackMethod(method, 
                                        ttp=self.initPopulationTTP,
                                    )

    def __call__(self, population_size):
        return self.__fn(population_size, **self._configs)

    def help(self):
        info = """InitPopulationRand:
    metoda: 'tsp';  config: None;\n"""
        print(info)

    # initPopulationTTP -------------------------------------
    def initPopulationTTP(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        tsp_population = []
        kp_population  = []
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga tsp_individ in genome
            tsp_individ = np.random.permutation(tsp_individ)
            kp_individ  = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            # add to population
            tsp_population.append(tsp_individ)
            kp_population.append(kp_individ)
        # cast to array
        tsp_population = np.array(tsp_population, dtype=np.int32)
        kp_population  = np.array(kp_population,  dtype=np.int32)
        return {"tsp":tsp_population, "kp":kp_population}
    # initPopulationTTP =====================================
