#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class MutateKPWithRepair(RootGA):
    """
    Wrapper around a KP mutation operator that automatically repairs
    overweight solutions using FastKPRepair.

    Expects base_mutator to have the usual KP signature:
        new_kp = base_mutator(parent1_kp, parent2_kp, offspring_kp, **kw)
    """

    def __init__(self, base_mutator, repair, W, dataset=None):
        super().__init__()
        self.base_mutator = base_mutator      # e.g. MutateMateiKP(...)
        self.repair       = repair            # FastKPRepair instance
        self.Wmax         = float(W)
        self.dataset      = dataset
        self.__name       = f"{base_mutator} + FastKPRepair"

        # you can override this in setParameters if you want
        self.base_rate = getattr(base_mutator, "rate", None)

    def __str__(self):
        return self.__name

    def help(self):
        print("MutateKPWithRepair(base_mutator + FastKPRepair)")

    def setParameters(self, **kw):
        """
        Forward parameters to base mutator and keep RootGA's behavior.
        """
        super().setParameters(**kw)
        if hasattr(self.base_mutator, "setParameters"):
            self.base_mutator.setParameters(**kw)

    # GA will call THIS method
    def __call__(self, parent1, parent2, offspring, **kw):
        """
        parent1, parent2, offspring are KP chromosomes (1D 0/1 arrays).
        We:
          1) call the original KP mutation
          2) if overweight, run FastKPRepair
        """
        # Normal KP mutation
        mutated = self.base_mutator(parent1, parent2, offspring, **kw)

        kp = np.array(mutated, copy=True, dtype=np.int32)
        # Quick weight check using repair's precomputed weights
        if np.dot(kp, self.repair.w) > self.Wmax + 1e-9:
            kp = self.repair(kp)

        return kp
