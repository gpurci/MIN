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
        name = ""
        # constante pentru setarea algoritmului
        GENERATIONS     = 500 # numarul de generatii
        POPULATION_SIZE = 100 # numarul populatiei
        GENOME_LENGTH   = 4 # numarul de alele
        MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
        CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
        SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
        ELITE_SIZE      = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor

    def __init__(self, name=""):
        RootGA.name = name

    def __str__(self):
        info = """name: {}
    GENERATIONS     = {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    ELITE_SIZE      = {}""".format(RootGA.name, RootGA.GENERATIONS, RootGA.POPULATION_SIZE, RootGA.GENOME_LENGTH, RootGA.MUTATION_RATE, 
                                    RootGA.CROSSOVER_RATE, RootGA.SELECT_RATE, RootGA.ELITE_SIZE)
        return info

    def setParameters(self, **kw):
        RootGA.POPULATION_SIZE = kw.get("POPULATION_SIZE", RootGA.POPULATION_SIZE)
        RootGA.MUTATION_RATE   = kw.get("MUTATION_RATE",   RootGA.MUTATION_RATE)
        RootGA.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE",  RootGA.CROSSOVER_RATE)
        RootGA.SELECT_RATE     = kw.get("SELECT_RATE", RootGA.SELECT_RATE)
        RootGA.GENERATIONS     = kw.get("GENERATIONS", RootGA.GENERATIONS)
        RootGA.ELITE_SIZE      = kw.get("ELITE_SIZE",  RootGA.ELITE_SIZE)
