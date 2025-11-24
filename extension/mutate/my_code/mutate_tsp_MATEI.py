#!/usr/bin/python

import numpy as np
from extension.mutate.my_code.mutate_base import *

class MutateMateiTSP(MutateBase):
    """
    TSP / permutation mutation operators (Matei combo).

    Methods:
        - 'swap'     : swap two random positions.
        - 'scramble' : scramble a random subsequence.
        - 'inversion': reverse a random subsequence.
        - 'mixt'     : mixture between the three above.

    Config:
        - subset_size: typical size of the mutated block (for scramble/inversion).
                       If missing, a small block is chosen randomly.
        - p_select   : probabilities for 'mixt' as [p_swap, p_scramble, p_inversion].
                       Defaults to [1/3, 1/3, 1/3].
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="MutateMateiTSP", **configs)
        self.__fn = self._unpackMethod(
            method,
            swap=self.mutateSwap,
            scramble=self.mutateScramble,
            inversion=self.mutateInversion,
            mixt=self.mutateMixt,
        )

    def __call__(self, parent1, parent2, offspring):
        """
        parent1, parent2: not used directly here, but kept for compatibility.
        offspring: 1D numpy array encoding a permutation.
        """
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateMateiTSP:
    metoda: 'swap';      config None;
    metoda: 'scramble';  config -> "subset_size":7, ;
    metoda: 'inversion'; config -> "subset_size":7, ;
    metoda: 'mixt';      config -> "subset_size":7, "p_select":[1/3,1/3,1/3], ;\n"""
        print(info)

    # -------------------------- SWAP ---------------------------------
    def mutateSwap(self, parent1, parent2, offspring, **kw):
        n = self.GENOME_LENGTH
        if n <= 1:
            return offspring.copy()

        child = offspring.copy()
        i, j = np.random.randint(0, n, size=2)
        if i == j:
            j = (j + 1) % n
        child[i], child[j] = child[j].copy(), child[i].copy()
        return child

    # ------------------------ SCRAMBLE -------------------------------
    def mutateScramble(self, parent1, parent2, offspring, subset_size=7, **kw):
        n = self.GENOME_LENGTH
        if n <= 2:
            return self.mutateSwap(parent1, parent2, offspring, **kw)

        subset_size = kw.get("subset_size", subset_size)
        subset_size = int(max(2, min(subset_size, n)))

        # choose random block of length subset_size
        start = np.random.randint(0, n - subset_size + 1)
        end = start + subset_size

        child = offspring.copy()
        block = child[start:end].copy()
        np.random.shuffle(block)
        child[start:end] = block
        return child

    # ------------------------ INVERSION ------------------------------
    def mutateInversion(self, parent1, parent2, offspring, subset_size=7, **kw):
        n = self.GENOME_LENGTH
        if n <= 2:
            return self.mutateSwap(parent1, parent2, offspring, **kw)

        subset_size = kw.get("subset_size", subset_size)
        subset_size = int(max(2, min(subset_size, n)))

        start = np.random.randint(0, n - subset_size + 1)
        end = start + subset_size

        child = offspring.copy()
        child[start:end] = child[start:end][::-1]
        return child

    # --------------------------- MIXTURE ------------------------------
    def mutateMixt(self, parent1, parent2, offspring, p_select=None, subset_size=7, **kw):
        """
        Apply one of [swap, scramble, inversion] according to p_select.
        """
        if p_select is None:
            p_select = self._configs.get("p_select", [1.0 / 3, 1.0 / 3, 1.0 / 3])

        try:
            p = np.array(p_select, dtype=float)
            if p.size != 3 or np.any(p < 0):
                raise ValueError
            s = p.sum()
            if s <= 0:
                raise ValueError
            p /= s
        except Exception:
            p = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=float)

        choice = np.random.choice([0, 1, 2], p=p)
        if choice == 0:
            return self.mutateSwap(parent1, parent2, offspring, **kw)
        elif choice == 1:
            return self.mutateScramble(parent1, parent2, offspring, subset_size=subset_size, **kw)
        else:
            return self.mutateInversion(parent1, parent2, offspring, subset_size=subset_size, **kw)

