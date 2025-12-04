#!/usr/bin/python

import numpy as np
from extension.ga_choice_base import *

class CrossoverChoice(GAChoiceBase):
    """
    """
    def __init__(self, *objects, p_select=None, scores=None):
        super().__init__(*objects, p_select=p_select, scores=scores, name="CrossoverChoice", inherit_class="CrossoverBase")

    def __call__(self, parent1, parent2):
        cond = np.random.choice(self._range, size=None, p=self_p_select)
        return self._objects[cond](parent1, parent2)
