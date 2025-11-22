#!/usr/bin/python

import numpy as np
from extension.ga_base import *

class StresBase(GABase):
    """
    Clasa 'StresBase', 
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="StresBase", **configs):
        super().__init__(method, name=name, **configs)
