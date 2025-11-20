#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

# =======================================================================
#   TSP / PERMUTATION MUTATION OPERATORS
# =======================================================================
class MutateTSP(RootGA):
    """
    Extension version for TSP / permutation chromosomes.

    Use in GA config:
        mutate_tsp = {
            "method": "extern",
            "extern_fn": MutateTSP("mixt"),
            "subset_size": 4,
            "p_method": [...]
        }
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__setMethods(method)

    def __str__(self):
        return f"MutateTSP(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """MutateTSP:
    Methods:
        'swap', 'swap_sim', 'swap_diff'
        'roll', 'roll_sim', 'roll_diff'
        'scramble', 'scramble_sim', 'scramble_diff'
        'inversion', 'inversion_sim', 'inversion_diff'
        'insertion'
        'mixt' -> uses p_method[] and subset_size"""

    # --------------------------------------------------------------
    #     INTERNAL SETUP
    # --------------------------------------------------------------
    def __setMethods(self, method):
        self.__method = method
        self.__fn = self.__resolve(method)

    def __resolve(self, method):
        fn = self.mutateAbstract
        if method is None:
            return fn

        table = {
            "swap": self.mutateSwap,
            "swap_sim": self.mutateSwapSim,
            "swap_diff": self.mutateSwapDiff,

            "roll": self.mutateRoll,
            "roll_sim": self.mutateRollSim,
            "roll_diff": self.mutateRollDiff,

            "scramble": self.mutateScramble,
            "scramble_sim": self.mutateScrambleSim,
            "scramble_diff": self.mutateScrambleDiff,

            "inversion": self.mutateInversion,
            "inversion_sim": self.mutateInversionSim,
            "inversion_diff": self.mutateInversionDiff,

            "insertion": self.mutateInsertion,
            "mixt": self.mutateMixt
        }

        return table.get(method, fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)

    def __call__(self, parent1, parent2, offspring, **call_cfg):
        tmp_cfg = self.__configs.copy()
        tmp_cfg.update(call_cfg)
        return self.__fn(parent1, parent2, offspring, **tmp_cfg)

    def mutateAbstract(self, parent1, parent2, offspring):
        raise NameError(f"No method '{self.__method}' configured in MutateTSP")

    # ==================================================================
    #   TSP / PERMUTATION OPERATORS
    # ==================================================================
    def mutateSwap(self, p1, p2, off):
        loc1, loc2 = np.random.randint(0, self.GENOME_LENGTH, size=2)
        off[loc1], off[loc2] = off[loc2], off[loc1]
        return off

    def mutateSwapSim(self, p1, p2, off):
        mask = (p1 == off) | (p2 == off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        loc1, loc2 = np.random.choice(args, size=2, replace=False)
        off[loc1], off[loc2] = off[loc2], off[loc1]
        return off

    def mutateSwapDiff(self, p1, p2, off):
        mask = (p1 != off) | (p2 != off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        loc1, loc2 = np.random.choice(args, size=2, replace=False)
        off[loc1], off[loc2] = off[loc2], off[loc1]
        return off

    def mutateRoll(self, p1, p2, off):
        shift = np.random.randint(0, self.GENOME_LENGTH)
        return np.roll(off, shift)

    def mutateRollSim(self, p1, p2, off):
        mask = (p1 == off) | (p2 == off)
        args = np.where(mask)[0]
        if len(args) == 0:
            return off
        start = np.random.choice(args)
        return np.roll(off, start)

    def mutateRollDiff(self, p1, p2, off):
        mask = (p1 != off) | (p2 != off)
        args = np.where(mask)[0]
        if len(args) == 0:
            return off
        start = np.random.choice(args)
        return np.roll(off, start)

    def mutateScramble(self, p1, p2, off, subset_size=4):
        idx = np.random.choice(self.GENOME_LENGTH, subset_size, replace=False)
        off[idx] = np.random.permutation(off[idx])
        return off

    def mutateScrambleSim(self, p1, p2, off, subset_size=4):
        mask = (p1 == off) | (p2 == off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        size = min(subset_size, len(args))
        idx = np.random.choice(args, size, replace=False)
        off[idx] = np.random.permutation(off[idx])
        return off

    def mutateScrambleDiff(self, p1, p2, off, subset_size=4):
        mask = (p1 != off) | (p2 != off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        size = min(subset_size, len(args))
        idx = np.random.choice(args, size, replace=False)
        off[idx] = np.random.permutation(off[idx])
        return off

    def mutateInversion(self, p1, p2, off):
        loc1, loc2 = sorted(np.random.randint(0, self.GENOME_LENGTH, size=2))
        off[loc1:loc2] = off[loc1:loc2][::-1]
        return off

    def mutateInversionSim(self, p1, p2, off):
        mask = (p1 == off) | (p2 == off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        loc1, loc2 = sorted(np.random.choice(args, 2, replace=False))
        off[loc1:loc2] = off[loc1:loc2][::-1]
        return off

    def mutateInversionDiff(self, p1, p2, off):
        mask = (p1 != off) | (p2 != off)
        args = np.where(mask)[0]
        if len(args) <= 1:
            return off
        loc1, loc2 = sorted(np.random.choice(args, 2, replace=False))
        off[loc1:loc2] = off[loc1:loc2][::-1]
        return off

    def mutateInsertion(self, p1, p2, off):
        loc1, loc2 = np.random.randint(0, self.GENOME_LENGTH, size=2)
        gene = off[loc2]
        off = np.delete(off, loc2)
        off = np.insert(off, loc1, gene)
        return off

    def mutateMixt(self, p1, p2, off, subset_size=4, p_method=None):
        op = np.random.choice(13, p=p_method)
        mapping = [
            self.mutateSwap, self.mutateSwapSim, self.mutateSwapDiff,
            self.mutateRoll, self.mutateRollSim, self.mutateRollDiff,
            self.mutateScramble, self.mutateScrambleSim, self.mutateScrambleDiff,
            self.mutateInversion, self.mutateInversionSim, self.mutateInversionDiff,
            self.mutateInsertion
        ]
        return mapping[op](p1, p2, off)
