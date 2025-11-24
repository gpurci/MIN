#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class CrossoverKPWithRepair(RootGA):
    """
    Wrapper around a KP crossover operator that automatically repairs
    overweight children using FastKPRepair.

    Expects base_crossover to have signature:
        child_kp = base_crossover(parent1_kp, parent2_kp, **kw)
    """

    def __init__(self, base_crossover, repair, W, dataset=None):
        super().__init__()
        self.base_crossover = base_crossover   # e.g. CrossoverMateiKP(...)
        self.repair         = repair           # FastKPRepair instance
        self.Wmax           = float(W)
        self.dataset        = dataset
        self.__name         = f"{base_crossover} + FastKPRepair"

    def __str__(self):
        return self.__name

    def help(self):
        print("CrossoverKPWithRepair(base_crossover + FastKPRepair)")

    def setParameters(self, **kw):
        """
        Forward parameters to base crossover and keep RootGA's behavior.
        """
        super().setParameters(**kw)
        if hasattr(self.base_crossover, "setParameters"):
            self.base_crossover.setParameters(**kw)

    # GA will call THIS method
    def __call__(self, parent1, parent2, **kw):
        """
        parent1, parent2 are KP chromosomes (1D 0/1 arrays).
        We:
          1) call the original KP crossover
          2) if overweight, run FastKPRepair
        """
        child = self.base_crossover(parent1, parent2, **kw)

        kp = np.array(child, copy=True, dtype=np.int32)
        if np.dot(kp, self.repair.w) > self.Wmax + 1e-9:
            kp = self.repair(kp)

        return kp
