#!/usr/bin/python

import numpy as np
from extern_fn import *

class Stres(ExtenFn):
    """
    Clasa 'Stres', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Stres")

    def __call__(self, genoms, scores):
        return self._extern_fn(genoms, scores)
