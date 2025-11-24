#!/usr/bin/python

class RootGA(object):
    """
    Clasa root pentru algoritmi genetici:
    In cadrul clasei root:
        - initializare variabile generale, pentru rularea algoritmului genetic
        - setare variabile generale
        - scurta descriere
    """
    def __init__(self):
        # constante pentru setarea algoritmului
        self.GENERATIONS     = 500 # numarul de generatii
        self.POPULATION_SIZE = 100 # numarul populatiei
        self.GENOME_LENGTH   = 8 # numarul de alele
        self.MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
        self.CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
        self.SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
        self.ELITE_SIZE      = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor

    def __str__(self):
        info = """
    GENERATIONS     = {}
    POPULATION_SIZE = {}
    GENOME_LENGTH   = {}
    MUTATION_RATE   = {}
    CROSSOVER_RATE  = {}
    SELECT_RATE     = {}
    ELITE_SIZE      = {}""".format(self.GENERATIONS, self.POPULATION_SIZE, self.GENOME_LENGTH, self.MUTATION_RATE, 
                                    self.CROSSOVER_RATE, self.SELECT_RATE, self.ELITE_SIZE)
        return info

    def setParameters(self, **kw):
        self.POPULATION_SIZE = kw.get("POPULATION_SIZE", self.POPULATION_SIZE)
        self.GENOME_LENGTH   = kw.get("GENOME_LENGTH",   self.GENOME_LENGTH)
        self.MUTATION_RATE   = kw.get("MUTATION_RATE",   self.MUTATION_RATE)
        self.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE",  self.CROSSOVER_RATE)
        self.SELECT_RATE     = kw.get("SELECT_RATE", self.SELECT_RATE)
        self.GENERATIONS     = kw.get("GENERATIONS", self.GENERATIONS)
        self.ELITE_SIZE      = kw.get("ELITE_SIZE",  self.ELITE_SIZE)
        # check
        error_msg = "Valoarea: 'POPULATION_SIZE': '{}', este mai mica decat: '1'".format(self.POPULATION_SIZE)
        assert (self.POPULATION_SIZE > 0), error_msg
        error_msg = "Valoarea: 'ELITE_SIZE': '{}', este mai mica decat: '1'".format(self.ELITE_SIZE)
        assert (self.ELITE_SIZE > 0), error_msg
        error_msg = "Valoarea: 'POPULATION_SIZE': '{}', este mai mica decat: 'ELITE_SIZE': '{}'".format(self.POPULATION_SIZE, self.ELITE_SIZE)
        assert (self.POPULATION_SIZE > self.ELITE_SIZE), error_msg
        error_msg = "Valoarea: 'GENOME_LENGTH': '{}', este mai mica decat: '1'".format(self.GENOME_LENGTH)
        assert (self.GENOME_LENGTH > 0), error_msg
        error_msg = "Valoarea: 'GENERATIONS': '{}', este mai mica decat: '0'".format(self.GENERATIONS)
        assert (self.GENERATIONS >= 0), error_msg

        error_msg = "Valoarea: 'MUTATION_RATE': '{}', este mai mica decat: '0'".format(self.MUTATION_RATE)
        assert (self.MUTATION_RATE >= 0), error_msg
        error_msg = "Valoarea: 'MUTATION_RATE': '{}', este mai mare decat: '1'".format(self.MUTATION_RATE)
        assert (self.MUTATION_RATE <= 1), error_msg

        error_msg = "Valoarea: 'CROSSOVER_RATE': '{}', este mai mica decat: '0'".format(self.CROSSOVER_RATE)
        assert (self.CROSSOVER_RATE >= 0), error_msg
        error_msg = "Valoarea: 'CROSSOVER_RATE': '{}', este mai mare decat: '1'".format(self.CROSSOVER_RATE)
        assert (self.CROSSOVER_RATE <= 1), error_msg

        error_msg = "Valoarea: 'SELECT_RATE': '{}', este mai mica decat: '0'".format(self.SELECT_RATE)
        assert (self.SELECT_RATE >= 0), error_msg
        error_msg = "Valoarea: 'SELECT_RATE': '{}', este mai mare decat: '1'".format(self.SELECT_RATE)
        assert (self.SELECT_RATE <= 1), error_msg

