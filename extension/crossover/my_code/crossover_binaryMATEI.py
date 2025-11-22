#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *
from extension.crossover.my_code.crossover_permMATEI import recSim

class CrossoverBinary(RootGA):
    """
    Crossover operators for binary chromosomes (KP).
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
        diff | split | perm_sim | flip_sim | uniform | one_point | two_point | mixt
        """

    def __unpackMethod(self, method):
        table = {
            "diff":      self.crossoverDiff,
            "split":     self.crossoverSplit,
            "perm_sim":  self.crossoverPermSim,
            "flip_sim":  self.crossoverFlipSim,
            "uniform":   self.uniform,
            "one_point": self.one_point,
            "two_point": self.two_point,
            "mixt":      self.crossoverMixt,
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

    # ----------------------------------------------------------------------
    #  CROSSOVERS FOR KP BINARY CHROMOSOME
    # ----------------------------------------------------------------------
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

            np.random.shuffle(union_genes)
            offspring[diff_locus] = union_genes

        return offspring

    def crossoverSplit(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        start, end = np.random.randint(0, self.GENOME_LENGTH, size=2)
        if start > end: start, end = end, start
        offspring[start:end] = parent2[start:end]
        return offspring

    def crossoverPermSim(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        mask_sim = parent1 == parent2
        sim_idx = np.argwhere(mask_sim)
        if sim_idx.shape[0] >= 4:
            start, length = recSim(mask_sim, 0, 0, 0)
            if length > 1:
                loc = np.arange(start, start+length)
                offspring[loc] = np.random.permutation(offspring[loc])
        return offspring

    def crossoverFlipSim(self, parent1, parent2, low, high):
        offspring = parent1.copy()
        mask_sim = parent1 == parent2
        sim_idx = np.argwhere(mask_sim)
        if sim_idx.shape[0] >= 4:
            start, length = recSim(mask_sim, 0, 0, 0)
            if length > 1:
                loc = np.arange(start, start+length)
                offspring[loc] = np.flip(offspring[loc])
        return offspring

    # ----------------------------------------------------------------------
    #  STANDARD GA CROSSOVERS
    # ----------------------------------------------------------------------

    def uniform(self, p1, p2, low=0, high=None, **kw):
        n = len(p1)
        mask = np.random.rand(n) < 0.5
        child = p1.copy()
        child[mask] = p2[mask]
        return child

    def one_point(self, p1, p2, low=0, high=None, **kw):
        n = len(p1)
        cut = np.random.randint(1, n-1)
        child = np.empty_like(p1)
        child[:cut] = p1[:cut]
        child[cut:] = p2[cut:]
        return child

    def two_point(self, p1, p2, low=0, high=None, **kw):
        n = len(p1)
        a, b = np.random.randint(1, n-1, size=2)
        if a > b: a, b = b, a
        child = p1.copy()
        child[a:b] = p2[a:b]
        return child

    # ----------------------------------------------------------------------
    # MIXT CROSSOVER
    # ----------------------------------------------------------------------

    def crossoverMixt(self, parent1, parent2, low, high, p_method=None):
        choice = np.random.choice([0,1,2,3], p=p_method)
        if choice == 0: return self.crossoverSplit(parent1, parent2, low, high)
        if choice == 1: return self.crossoverDiff(parent1, parent2, low, high)
        if choice == 2: return self.crossoverPermSim(parent1, parent2, low, high)
        return self.crossoverFlipSim(parent1, parent2, low, high)
