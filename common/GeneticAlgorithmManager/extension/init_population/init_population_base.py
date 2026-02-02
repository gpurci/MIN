#!/usr/bin/python

import numpy as np
from extension.ga_base import *

class InitPopulationBase(GABase):
    """
    Clasa 'InitPopulationBase', 
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="InitPopulationBase", **configs):
        super().__init__(method, name=name, **configs)
