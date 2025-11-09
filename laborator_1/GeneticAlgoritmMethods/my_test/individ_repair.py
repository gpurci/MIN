#!/usr/bin/python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from laborator_1.GeneticAlgoritmMethods.my_code.root_GA import RootGA

class TestIndividRepair(RootGA):
    """
    Clasa 'IndividRepair', ofera doar metode pentru a initializa populatia.
    Functia 'individRepair' are 1 parametru, individ - individul care va fi reparat.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__(config)

    def test(self, config):
        pass
        # test implementation
