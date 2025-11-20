#!/usr/bin/python

import numpy as np
from extern_fn import *

class Fitness(ExtenFn):
    """
    Clasa 'Fitness', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Fitness")

    def __call__(self, metric_values):
        return self.__extern_fn(metric_values)
