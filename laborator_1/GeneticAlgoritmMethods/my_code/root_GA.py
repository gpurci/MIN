#!/usr/bin/python

class RootGA(object):
    """
    Clasa root pentru algoritmi genetici.

    În cadrul clasei:
    - inițializare variabile generale necesare rulării algoritmului genetic
    - setarea parametrilor globali
    - descriere succintă
    """
    def __init__(self):
        # constante de configurare GA
        self.GENERATIONS = 500          # numarul de generatii
        self.POPULATION_SIZE = 100      # numarul populatiei
        self.GENOME_LENGTH = 8          # numarul de alele
        self.MUTATION_RATE = 0.01       # probabilitatea de mutatie
        self.CROSSOVER_RATE = 0.5       # probabilitatea de incrucisare
        self.SELECT_RATE = 0.8          # probabilitate la selectie
        self.ELITE_SIZE = 5             # numarul individilor de elită

    def __str__(self):
        return (
            f"GENERATIONS = {self.GENERATIONS}\n"
            f"POPULATION_SIZE = {self.POPULATION_SIZE}\n"
            f"GENOME_LENGTH = {self.GENOME_LENGTH}\n"
            f"MUTATION_RATE = {self.MUTATION_RATE}\n"
            f"CROSSOVER_RATE = {self.CROSSOVER_RATE}\n"
            f"SELECT_RATE = {self.SELECT_RATE}\n"
            f"ELITE_SIZE = {self.ELITE_SIZE}"
        )

    def setParameters(self, **kw):
        """
        Setează parametrii globali ai algoritmului genetic.
        Dacă o cheie nu este furnizată, păstrează valoarea anterioară.
        """

        self.POPULATION_SIZE = kw.get("POPULATION_SIZE", self.POPULATION_SIZE)
        self.GENOME_LENGTH = kw.get("GENOME_LENGTH", self.GENOME_LENGTH)
        self.MUTATION_RATE = kw.get("MUTATION_RATE", self.MUTATION_RATE)
        self.CROSSOVER_RATE = kw.get("CROSSOVER_RATE", self.CROSSOVER_RATE)
        self.SELECT_RATE = kw.get("SELECT_RATE", self.SELECT_RATE)
        self.GENERATIONS = kw.get("GENERATIONS", self.GENERATIONS)
        self.ELITE_SIZE = kw.get("ELITE_SIZE", self.ELITE_SIZE)
