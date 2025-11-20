#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

# =======================================================================
#   KP / BINARY MUTATION OPERATORS
# =======================================================================
class MutateKP(RootGA):
    """
    Extension version for KP / binary chromosomes.
    Only binary mutations are included (no TSP operators).

    Use in GA config:
        mutate_kp = {
            "method": "extern",
            "extern_fn": MutateKP("mixt_binary"),
            "p_method": [...]
        }
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__setMethods(method)

    def __str__(self):
        return f"MutateKP(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """MutateKP:
    Methods:
        'binary'       -> swap two bits
        'binary_sim'   -> swap bits that are same as parents
        'binary_diff'  -> swap bits that differ from parents
        'flip'         -> flip a random bit
        'flip_sim'     -> flip bit similar to parents
        'flip_diff'    -> flip bit different from parents
        'mixt_binary'  -> mixture (uses p_method[])"""

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
            "binary": self.mutateBinary,
            "binary_sim": self.mutateBinarySim,
            "binary_diff": self.mutateBinaryDiff,

            "flip": self.mutateFlip,
            "flip_sim": self.mutateFlipSim,
            "flip_diff": self.mutateFlipDiff,

            "mixt_binary": self.mutateMixtBinary
        }

        return table.get(method, fn)

    def __call__(self, p1, p2, off, **kw):
        return self.__fn(p1, p2, off, **{**self.__configs, **kw})

    def mutateAbstract(self, p1, p2, off):
        raise NameError(f"No method '{self.__method}' configured in MutateKP")

    # ==================================================================
    #   PURE KP / BINARY OPERATORS
    # ==================================================================
    def mutateBinary(self, p1, p2, off):
        """Swap two random bits"""
        i, j = np.random.randint(0, self.GENOME_LENGTH, 2)
        off[i], off[j] = off[j], off[i]
        return off

    def mutateBinarySim(self, p1, p2, off):
        mask = (p1 == off) | (p2 == off)
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            return off
        i, j = np.random.choice(idx, 2, replace=False)
        off[i], off[j] = off[j], off[i]
        return off

    def mutateBinaryDiff(self, p1, p2, off):
        mask = (p1 != off) | (p2 != off)
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            return off
        i, j = np.random.choice(idx, 2, replace=False)
        off[i], off[j] = off[j], off[i]
        return off

    # ----------------
    # BIT FLIPS
    # ----------------
    def mutateFlip(self, p1, p2, off):
        loc = np.random.randint(0, self.GENOME_LENGTH)
        off[loc] = 1 - off[loc]
        return off

    def mutateFlipSim(self, p1, p2, off):
        mask = (p1 == off) | (p2 == off)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return off
        loc = np.random.choice(idx)
        off[loc] = 1 - off[loc]
        return off

    def mutateFlipDiff(self, p1, p2, off):
        mask = (p1 != off) | (p2 != off)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return off
        loc = np.random.choice(idx)
        off[loc] = 1 - off[loc]
        return off

    # ----------------
    # MIXED KP OPERATOR
    # ----------------
    def mutateMixtBinary(self, p1, p2, off, p_method=None):
        op = np.random.choice(6, p=p_method)
        ops = [
            self.mutateBinary,
            self.mutateBinarySim,
            self.mutateBinaryDiff,
            self.mutateFlip,
            self.mutateFlipSim,
            self.mutateFlipDiff
        ]
        return ops[op](p1, p2, off)
