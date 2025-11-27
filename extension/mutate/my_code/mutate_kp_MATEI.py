#!/usr/bin/python
import numpy as np

from extension.mutate.my_code.mutate_base import *


class MutateMateiKP(MutateBase):
    """
    Knapsack mutation for TTP with one-bit-per-city representation.

    Invariants it enforces:
      - len(kp) == GENOME_LENGTH
      - city i ↔ bit kp[i]
      - city 0 has no item => kp[0] is always forced to 0

    Methods (for the GA 'method' argument):
      - 'bitflip'        -> mutateBitflip
      - 'single_bit'     -> mutateSingleBit
      - 'improved'       -> mutateImproved (flip + repair + greedy fill)
      - 'mixt'           -> mixture of (bitflip, single_bit)
      - 'mixt_extended'  -> mixture of (bitflip, single_bit, improved)
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="MutateMateiKP", **configs)

        self.__fn = self._unpackMethod(
            method,
            bitflip=self.mutateBitflip,
            single_bit=self.mutateSingleBit,
            improved=self.mutateImproved,
            mixt=self.mutateMixt,
            mixt_extended=self.mutateMixtExtended,
        )

        # dataset, capacity W are attached later via setParameters(...)
        self.dataset = configs.get("dataset", None)
        self.W = configs.get("W", None)

        self._rng = np.random.RandomState()
        self._configs = dict(configs)

    # =======================================================
    # Allow dataset / W to be attached by GA
    # =======================================================
    def setParameters(self, **kw):
        super().setParameters(**kw)

        if "dataset" in kw:
            self.dataset = kw["dataset"]

        if "W" in kw:
            self.W = kw["W"]

    # =======================================================
    # Main call
    # =======================================================
    def __call__(self, parent1, parent2, offspring):
        """
        GA interface: offspring is a 1-D KP bitstring.
        """
        child = self.__fn(parent1, parent2, offspring, **self._configs)
        # enforce depot empty
        if child.shape[0] > 0:
            child[0] = 0
        return child

    # =======================================================
    # Improved KP mutation (TTP aware through capacity and ratio)
    # =======================================================
    def mutateImproved(self, parent1, parent2, offspring, rate=0.02, **kw):
        """
        1. Flip a few bits (excluding city 0)
        2. If overweight → remove worst-density items
        3. If underweight → greedily add best-density items
        """

        x = offspring.copy()
        n = self.GENOME_LENGTH
        if n == 0:
            return x

        if self.dataset is None or self.W is None:
            # fall back gracefully if GA forgot to attach dataset/W
            return self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)

        w = np.asarray(self.dataset["item_weight"], dtype=np.float64)
        v = np.asarray(self.dataset["item_profit"], dtype=np.float64)
        Wmax = float(self.W)

        eps = 1e-9
        ratio = v / (w + eps)
        ratio[w <= 0] = 0.0   # city 0 or any dummy items are neutral

        rate = float(kw.get("rate", rate))
        rate = max(0.0, min(rate, 1.0))

        # -------- Step 1: controlled random flips (excluding depot) --------
        if n <= 1:
            x[:] = 0
            return x

        idx_candidates = np.arange(1, n)  # never touch city 0
        k = max(1, int(idx_candidates.size * rate))
        k = min(k, idx_candidates.size)

        flip_idx = self._rng.choice(idx_candidates, size=k, replace=False)
        x[flip_idx] ^= 1

        # depot must always be empty
        x[0] = 0

        # -------- Step 2: overweight repair via worst-density removal --------
        total_w = float(np.dot(x, w))

        if total_w > Wmax:
            ones = np.where(x == 1)[0]
            ones = ones[ones != 0]  # exclude depot

            # remove lowest density first
            order = np.argsort(ratio[ones])  # worst first
            for j in ones[order]:
                x[j] = 0
                total_w -= w[j]
                if total_w <= Wmax:
                    break

        else:
            # -------- Step 3: greedy fill with best density items --------
            zeros = np.where(x == 0)[0]
            zeros = zeros[zeros != 0]  # exclude depot
            if zeros.size > 0:
                best_order = zeros[np.argsort(-ratio[zeros])]  # best first
                for j in best_order:
                    if total_w + w[j] <= Wmax:
                        x[j] = 1
                        total_w += w[j]

        # final safety: depot empty
        x[0] = 0
        return x

    # =======================================================
    # Bitflip
    # =======================================================
    def mutateBitflip(self, parent1, parent2, offspring, rate=0.02, **kw):
        n = self.GENOME_LENGTH
        if n == 0:
            return offspring.copy()

        rate = float(kw.get("rate", rate))
        rate = max(0.0, min(rate, 1.0))

        child = offspring.copy()
        mask = self._rng.rand(n) < rate
        # never flip depot position
        if n > 0:
            mask[0] = False

        child[mask] ^= 1
        if n > 0:
            child[0] = 0
        return child

    # =======================================================
    # Single bit flip
    # =======================================================
    def mutateSingleBit(self, parent1, parent2, offspring, **kw):
        n = self.GENOME_LENGTH
        if n <= 1:
            return offspring.copy()

        child = offspring.copy()
        # choose from cities 1..n-1 only (never depot)
        i = self._rng.randint(1, n)
        child[i] ^= 1
        child[0] = 0
        return child

    # =======================================================
    # Mixture: bitflip + single bit
    # =======================================================
    def mutateMixt(self, parent1, parent2, offspring,
                   p_select=None, rate=0.02, **kw):

        if p_select is None:
            p_select = self._configs.get("p_select", [0.7, 0.3])

        p = np.asarray(p_select, dtype=float)
        p /= p.sum()

        choice = self._rng.choice([0, 1], p=p)
        if choice == 0:
            child = self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)
        else:
            child = self.mutateSingleBit(parent1, parent2, offspring, **kw)

        if child.shape[0] > 0:
            child[0] = 0
        return child

    # =======================================================
    # Mixture: bitflip + single bit + improved
    # =======================================================
    def mutateMixtExtended(self, parent1, parent2, offspring,
                           p_select=None, rate=0.02, **kw):

        if p_select is None:
            p_select = self._configs.get("p_select", [0.4, 0.3, 0.3])

        p = np.asarray(p_select, dtype=float)
        p /= p.sum()

        choice = self._rng.choice([0, 1, 2], p=p)

        if choice == 0:
            child = self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)
        elif choice == 1:
            child = self.mutateSingleBit(parent1, parent2, offspring, **kw)
        else:
            child = self.mutateImproved(parent1, parent2, offspring, rate=rate, **kw)

        if child.shape[0] > 0:
            child[0] = 0
        return child
