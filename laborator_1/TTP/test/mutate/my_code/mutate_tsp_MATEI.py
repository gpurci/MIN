#!/usr/bin/python

import numpy as np
from extension.mutate.my_code.mutate_base import *
from extension.local_search_algorithms.two_opt import TwoOpt


class MutateMateiTSP(MutateBase):
    """
    TSP / permutation mutation operators (Matei combo) + light 2-opt local search.

    Methods:
        - 'swap'     : swap two random positions.
        - 'scramble' : scramble a random subsequence.
        - 'inversion': reverse a random subsequence.
        - 'mixt'     : mixture between the three above.

    Config:
        - subset_size: typical size of the mutated block (for scramble/inversion).
        - p_select   : probabilities for 'mixt' as [p_swap, p_scramble, p_inversion].
    """

    def __init__(self, method, dataset=None, **configs):
        # MutateBase handles method, configs, etc.
        super().__init__(method, name="MutateMateiTSP", **configs)

        # base mutation function (swap / scramble / inversion / mixt)
        self.__fn = self._unpackMethod(
            method,
            swap=self.mutateSwap,
            scramble=self.mutateScramble,
            inversion=self.mutateInversion,
            mixt=self.mutateMixt,
        )

        self._rng = np.random.RandomState()

        # ---- light 2-opt local search (TTP-style) ----
        # TwoOpt understands method="two_opt_LS" (as in TTPVNDLocalSearch)
        self._two_opt = None
        if dataset is not None:
            self._two_opt = TwoOpt(method="two_opt_LS", dataset=dataset)

    # ------------------------------------------------------------------
    # small 2-opt noise – always applied for diversity
    # ------------------------------------------------------------------
    def _two_opt_noise(self, route, prob=0.12):
        """
        Tiny stochastic 2-opt-like shuffle.
        Does NOT try to improve — only adds diversity.
        """
        rng = self._rng
        if rng.rand() > prob:
            return

        L = route.shape[0]
        if L < 4:
            return

        a, b = sorted(rng.choice(L, size=2, replace=False))
        if b - a <= 1:
            return

        route[a:b] = route[a:b][::-1]

    # ------------------------------------------------------------------
    # GA entry point (chromosome mode): tsp_child = mutator(p1_tsp, p2_tsp, tsp_child)
    # ------------------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **kw):
        """
        parent1, parent2, offspring are TSP chromosomes (1D permutations).
        MutateChromosome in GA passes exactly these.
        """
        # 1) normal permutation mutation (swap/scramble/inversion/mixt)
        child = self.__fn(parent1, parent2, offspring, **self._configs)

        # 2) occasionally do *real* 2-opt improvement (10% of time)
        if self._two_opt is not None and self._rng.rand() < 0.10:
            # TwoOpt supports call: two_opt(parent1, parent2, route)
            child = self._two_opt(parent1, parent2, child)

        # 3) always add a tiny 2-opt style noise for diversity
        self._two_opt_noise(child, prob=0.12)

        return child

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
    def mutateMixt(self, parent1, parent2, offspring,
                   p_select=None, subset_size=7, **kw):
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
            return self.mutateScramble(parent1, parent2, offspring,
                                       subset_size=subset_size, **kw)
        else:
            return self.mutateInversion(parent1, parent2, offspring,
                                        subset_size=subset_size, **kw)
