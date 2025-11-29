#!/usr/bin/python

import numpy as np

'''from sys_function import sys_remove_modules

sys_remove_modules("extern_fn")'''
from extern_fn import *

class InitPopulation(ExtenFn):
    """
    Clasa 'InitPopulation',
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "InitPopulation")

    def __call__(self, size, genoms):
        ret = self._extern_fn(size, genoms=genoms)

        if ret is None:
            # extern_fn did in-place init on genoms
            return genoms.population()
        else:
            # extern_fn returned a dict of chromosomes
            return genoms.concatChromosomes(**ret)

