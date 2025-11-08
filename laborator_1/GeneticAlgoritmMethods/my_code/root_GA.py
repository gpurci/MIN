#!/usr/bin/python

import numpy as np

class RootGA(object):
    """
    Clasa root pentru algoritmi genetici:
    In cadrul clasei root:
        - initializare variabile generale, pentru rularea algoritmului genetic
        - setare variabile generale
        - scurta descriere
    """
    def __init__(self, name=""):
        self.NAME = name
        # constante pentru setarea algoritmului
        self.GENERATIONS     = 500 # numarul de generatii
        self.POPULATION_SIZE = 100 # numarul populatiei
        self.GENOME_LENGTH   = 8 # numarul de alele
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
    ELITE_SIZE      = {}""".format(self.NAME, self.GENERATIONS, self.POPULATION_SIZE, self.GENOME_LENGTH, self.MUTATION_RATE, 
                                    self.CROSSOVER_RATE, self.SELECT_RATE, self.ELITE_SIZE)
        return info

    def setParameters(self, **kw):
        self.POPULATION_SIZE = kw.get("POPULATION_SIZE", self.POPULATION_SIZE)
        self.MUTATION_RATE   = kw.get("MUTATION_RATE",   self.MUTATION_RATE)
        self.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE",  self.CROSSOVER_RATE)
        self.SELECT_RATE     = kw.get("SELECT_RATE", self.SELECT_RATE)
        self.GENERATIONS     = kw.get("GENERATIONS", self.GENERATIONS)
        self.ELITE_SIZE      = kw.get("ELITE_SIZE",  self.ELITE_SIZE)

