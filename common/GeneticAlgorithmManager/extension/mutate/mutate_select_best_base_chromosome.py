#!/usr/bin/python

import numpy as np
from extension.ga_select_best_base import *

class MutateSelectBestChromosome(GASelectBestBase):
    """
    """
    def __init__(self, best_fn, *objects):
        super().__init__(*objects, name="MutateSelectBest", inherit_class="MutateBase")
        self.best_fn = best_fn

    def __call__(self, parent1, parent2, offspring):
        cond = np.random.choice(self._range, size=None, p=self_p_select)
        size = len(self)
        new_offspring = np.repeat(offspring.reshape(1, -1), size+1, axis=0)
        for idx in range(size):
            new_offspring[idx] = self._objects[idx](parent1, parent2, offspring)
        best_idx = self.best_fn(new_offspring)
        return 
