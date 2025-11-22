#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *


class InitRandomPopulation(RootGA):
    """
    Extension: Only RANDOM initial population generators.
    Compatible with:
        method = "TSP_rand"
        method = "TTP_rand"
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__setMethods(method)

    def __str__(self):
        info = """InitRandomPopulation:
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def help(self):
        info = """InitRandomPopulation:
    metoda: 'TSP_rand'; config: None;
    metoda: 'TTP_rand'; config: None;\n"""
        print(info)

    def __setMethods(self, method):
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __unpackMethod(self, method):
        fn = self.initPopulationAbstract
        if method == "TSP_rand":
            fn = self.initPopTSPRand
        elif method == "TTP_rand":
            fn = self.initPopTTPRand
        return fn

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self.__configs)

    # ABSTRACT -----------------------------------------------------
    def initPopulationAbstract(self, population_size, genoms=None, **kw):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru InitRandomPopulation."
        )

    # RANDOM TSP ---------------------------------------------------
    def initPopTSPRand(self, population_size=-1, genoms=None):
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)

        for _ in range(population_size):
            genoms.add(tsp=np.random.permutation(individ))

        genoms.saveInit()
        print("Random TSP population =", genoms.shape)

    # RANDOM TTP ---------------------------------------------------
    def initPopTTPRand(self, population_size=-1, genoms=None):
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)

        for _ in range(population_size):
            kp = np.random.randint(0, 2, size=self.GENOME_LENGTH)
            genoms.add(tsp=np.random.permutation(tsp_individ), kp=kp)

        genoms.saveInit()
        print("Random TTP population =", genoms.shape)
