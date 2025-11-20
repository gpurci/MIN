#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *
from extension.crossover.my_code.crossover_permMATEI import recSim

class CrossoverBinary(RootGA):
    """
    Crossover operators for binary chromosomes (KP).
    Same bodies as manager (unchanged), split for cleanliness.
    """

    def __init__(self, method=None, **kw):
        super().__init__()
        self.__configs = kw
        self.__method  = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"CrossoverBinary(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """CrossoverBinary:
    diff | split | perm_sim | flip_sim | mixt
"""

    def __unpackMethod(self, method):
        table = {
            "diff":     self.crossoverDiff,
            "split":    self.crossoverSplit,
            "perm_sim": self.crossoverPermSim,
            "flip_sim": self.crossoverFlipSim,
            "mixt":     self.crossoverMixt,
        }
        return table.get(method, self.crossoverAbstract)

    def setParameters(self, **kw):
        super().setParameters(**kw)

    def __call__(self, parent1, parent2, low=0, high=None, **call_configs):
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(parent1, parent2, low, high, **cfg)

    def crossoverAbstract(self, *a, **kw):
        raise NameError(f"Lipseste metoda '{self.__method}' pentru CrossoverBinary")

    def crossoverDiff(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        diff_locus = parent1 != parent2
        size_diff_locus = diff_locus.sum()

        if size_diff_locus >= 4:
            diff_genes1 = parent1[diff_locus]
            diff_genes2 = parent2[diff_locus]

            union_genes = np.union1d(diff_genes1, diff_genes2)

            size_needed = size_diff_locus - union_genes.shape[0]
            if size_needed > 0:
                new_genes = np.random.randint(low=low, high=high, size=size_needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif size_needed < 0:
                union_genes = union_genes[:size_needed]

            union_genes = np.random.permutation(union_genes)
            offspring[diff_locus] = union_genes

        return offspring

    def crossoverSplit(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        start, end = np.random.randint(0, self.GENOME_LENGTH, size=2)
        if start > end:
            start, end = end, start
        offspring[start:end] = parent2[start:end]
        return offspring

    def crossoverPermSim(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        mask_sim_locus = parent1 == parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if size >= 4:
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if lenght > 1:
                locuses = np.arange(start, start + lenght)
                offspring[locuses] = np.random.permutation(offspring[locuses])
        return offspring

    def crossoverFlipSim(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        mask_sim_locus = parent1 == parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if size >= 4:
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if lenght > 1:
                locuses = np.arange(start, start + lenght)
                offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def crossoverMixt(self, parent1, parent2, low, high, p_method=None):
        cond = np.random.choice([0, 1, 2, 3], p=p_method)
        if cond == 0:
            return self.crossoverSplit(parent1, parent2, low, high)
        if cond == 1:
            return self.crossoverDiff(parent1, parent2, low, high)
        if cond == 2:
            return self.crossoverPermSim(parent1, parent2, low, high)
        return self.crossoverFlipSim(parent1, parent2, low, high)
