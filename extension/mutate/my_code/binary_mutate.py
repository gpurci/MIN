#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class BinaryMutate(RootGA):
    """
    Binary mutation operators (KP chromosome) — with optional weight-safe repair.
    """

    def __init__(self, method, **configs):
        super().__init__()
        self.__method  = method
        self.__configs = configs
        self.__fn = self.__unpackMethod(method)

        # (Needed for weight safe repair)
        self.weight = None
        self.profit = None
        self.Wmax   = None

    def setParameters(self, **kw):
        super().setParameters(**kw)

        # GA will inject these through setParameters(...)
        self.weight = kw.get("weight", self.weight)
        self.profit = kw.get("profit", self.profit)
        self.Wmax   = kw.get("W", self.Wmax)

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self.__configs)

    def __str__(self):
        info  = "BinaryMutate: method '{}'\n".format(self.__method)
        tmp   = "configs: '{}'\n".format(self.__configs)
        info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method):
        fn = self.mutateAbstract
        if method is not None:
            if   method == "binary":      fn = self.mutateBinary
            elif method == "binary_sim":  fn = self.mutateBinarySim
            elif method == "binary_diff": fn = self.mutateBinaryDiff
            elif method == "mixt_binary": fn = self.mutateMixtBinary
        return fn

    def help(self):
        info = """BinaryMutate:
    'binary'        – flip 1 random bit (now weight-safe)
    'binary_sim'    – mutate inside longest similarity block
    'binary_diff'   – mutate inside longest difference block
    'mixt_binary'   – random choice between operators
"""
        print(info)

    def mutateAbstract(self, parent1, parent2, offspring):
        error_message = (
            f"Functia 'BinaryMutate', lipseste metoda '{self.__method}', config: '{self.__configs}'"
        )
        raise NameError(error_message)

    def mutateBinary(self, parent1, parent2, offspring):
        """Flip one random bit + repair overweight."""
        locus = np.random.randint(0, self.GENOME_LENGTH)
        offspring[locus] = 1 - offspring[locus]

        # If weight information not provided → do classic mutation only
        if self.Wmax is None or self.weight is None:
            return offspring

        # ----- compute current weight -----
        Wcur = np.dot(offspring, self.weight)

        if Wcur <= self.Wmax:
            return offspring

        # ----- OVERWEIGHT → REMOVE WORST ITEMS -----
        overweight = Wcur - self.Wmax
        idx_taken = np.where(offspring == 1)[0]

        if idx_taken.size == 0:
            return offspring

        eff = self.profit[idx_taken] / (self.weight[idx_taken] + 1e-9)
        order = np.argsort(eff)  # remove worst first

        removed_weight = 0
        for j in order:
            idx = idx_taken[j]
            offspring[idx] = 0
            removed_weight += self.weight[idx]
            if removed_weight >= overweight:
                break

        return offspring

    def mutateBinarySim(self, parent1, parent2, offspring):
        mask_locus = parent1 == parent2
        size = mask_locus.sum()

        if size >= 4:
            start, length = recSim(mask_locus, 0, 0, 0)
            if length > 1:
                locus = np.random.randint(start, start+length)
                offspring[locus] = 1 - offspring[locus]

        return offspring

    def mutateBinaryDiff(self, parent1, parent2, offspring):
        mask = parent1 != parent2
        size = mask.sum()

        if size >= 4:
            start, length = recSim(mask, 0, 0, 0)
            if length > 1:
                locus = np.random.randint(start, start+length)
                offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateMixtBinary(self, parent1, parent2, offspring, p_method=None):
        cond = np.random.choice([0,1,2], p=p_method)

        if cond == 0:
            return self.mutateBinary(parent1, parent2, offspring)
        elif cond == 1:
            return self.mutateBinarySim(parent1, parent2, offspring)
        else:
            return self.mutateBinaryDiff(parent1, parent2, offspring)


def recSim(mask_genes, start, lenght, arg):
    if arg < mask_genes.shape[0]:
        tmp_arg = arg
        tmp_st  = arg
        tmp_len = 0
        while tmp_arg < mask_genes.shape[0]:
            if mask_genes[tmp_arg]:
                tmp_arg += 1
            else:
                tmp_len = tmp_arg - tmp_st
                if lenght < tmp_len:
                    start, lenght = tmp_st, tmp_len
                return recSim(mask_genes, start, lenght, tmp_arg+1)
    else:
        return start, lenght
    return start, lenght
