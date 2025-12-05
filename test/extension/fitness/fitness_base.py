#!/usr/bin/python

import numpy as np
from extension.ga_base import *

class FitnessBase(GABase):
    """
    Clasa 'FitnessBase', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="FitnessBase", **configs):
        super().__init__(method, name=name, **configs)
