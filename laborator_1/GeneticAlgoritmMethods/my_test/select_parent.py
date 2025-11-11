#!/usr/bin/python

import numpy as np
from my_code.select_parent import *
from my_code.root_GA import *

class TestSelectParent(RootGA):
    """
    Clasa 'SelectParent', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__()
        self.selector = SelectParent(config)

    def test(self, config):
        print("\n=== Test SelectParent:", config, "===")

        # small fake 4-individual population fitness
        fitness = np.array([0.1, 0.2, 0.5, 0.2], dtype=np.float32)

        # configure strategy
        self.selector.setConfig(config)

        # must call startEpoch BEFORE selecting
        self.selector.startEpoch(fitness)

        # sample 20 picks
        picks = [int(self.selector()) for _ in range(20)]
        print("picks =", picks)

        # quick sanity check: each pick from {0,1,2,3}
        if all(0 <= p < 4 for p in picks):
            print("OK: all picks in range 0-3")
        else:
            print("ERROR: picks outside range")
