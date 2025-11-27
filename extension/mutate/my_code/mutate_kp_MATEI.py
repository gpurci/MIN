#!/usr/bin/python
import numpy as np

from extension.mutate.my_code.mutate_base import *


class MutateMateiKP(MutateBase):
    """
    Knapsack mutation for TTP with one-bit-per-city representation.

    Invariants it enforces:
      - len(kp) == GENOME_LENGTH
      - city i ↔ bit kp[i]
      - city 0 has no item => kp[0] is always 0

    Methods:
      - bitflip
      - single_bit
      - improved
      - mixt
      - mixt_extended
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

        # dataset, capacity W will be attached later via setParameters(...)
        self.dataset = configs.get("dataset", None)
        self.W = configs.get("W", None)

        # ============== NEW: soft capacity/light-speed tuning ============
        # soft_capacity_ratio < 1.0 = drop worst items even when under W
        self.soft_capacity_ratio = float(configs.get("soft_capacity_ratio", 1.0))
        self.max_soft_drop = int(configs.get("max_soft_drop", 3))

        # If dataset already given we pre-cache weights/profits
        if self.dataset is not None:
            self.item_weight = np.asarray(self.dataset["item_weight"], dtype=np.float64)
            self.item_profit = np.asarray(self.dataset["item_profit"], dtype=np.float64)
        else:
            self.item_weight = None
            self.item_profit = None

        self._rng = np.random.RandomState()
        self._configs = dict(configs)

    # =======================================================
    # Allow dataset / W to be attached by GA
    # =======================================================
    def setParameters(self, **kw):
        super().setParameters(**kw)

        if "dataset" in kw:
            self.dataset = kw["dataset"]
            self.item_weight = np.asarray(self.dataset["item_weight"], dtype=np.float64)
            self.item_profit = np.asarray(self.dataset["item_profit"], dtype=np.float64)

        if "W" in kw:
            self.W = kw["W"]

        if "soft_capacity_ratio" in kw:
            self.soft_capacity_ratio = float(kw["soft_capacity_ratio"])
        if "max_soft_drop" in kw:
            self.max_soft_drop = int(kw["max_soft_drop"])

    # =======================================================
    # Main GA call
    # =======================================================
    def __call__(self, parent1, parent2, offspring):
        child = self.__fn(parent1, parent2, offspring, **self._configs)
        if child.shape[0] > 0:
            child[0] = 0
        return child

    # =======================================================
    # Soft Speed-Aware "Lightening" (NEW)
    # =======================================================
    def _lighten_for_speed(self, x):
        """
        If total weight > soft_W, drop worst-density items to improve speed,
        even when x is not overweight.
        """
        if (
            self.soft_capacity_ratio >= 1.0
            or self.dataset is None
            or self.W is None
            or self.item_weight is None
        ):
            return x

        w = self.item_weight
        p = self.item_profit
        Wmax = float(self.W)

        total_w = float(np.dot(x, w))
        soft_W = self.soft_capacity_ratio * Wmax

        if total_w <= soft_W:
            return x

        # compute densities
        density = p / (w + 1e-12)
        density = density.copy()
        density[x == 0] = np.inf  # don't evaluate unpicked items

        for _ in range(self.max_soft_drop):
            if total_w <= soft_W:
                break

            j = int(np.argmin(density))
            if not x[j]:
                break
            x[j] = 0
            total_w -= w[j]
            density[j] = np.inf

        return x

    # =======================================================
    # Improved KP mutation (MAIN MODE)
    # =======================================================
    def mutateImproved(self, parent1, parent2, offspring, rate=0.02, **kw):
        x = offspring.copy()
        n = self.GENOME_LENGTH
        if n <= 1:
            return x

        if self.dataset is None or self.W is None:
            return self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)

        w = self.item_weight
        v = self.item_profit
        Wmax = float(self.W)

        eps = 1e-9
        ratio = v / (w + eps)
        ratio[w <= 0] = 0.0

        # Step 1 — random flips
        idx = np.arange(1, n)
        k = max(1, int(len(idx) * rate))
        k = min(k, len(idx))
        flip_idx = self._rng.choice(idx, size=k, replace=False)
        x[flip_idx] ^= 1
        x[0] = 0

        # Step 2 — hard overweight fix
        total_w = float(np.dot(x, w))
        if total_w > Wmax:
            ones = np.where(x == 1)[0]
            ones = ones[ones != 0]
            order = np.argsort(ratio[ones])
            for j in ones[order]:
                x[j] = 0
                total_w -= w[j]
                if total_w <= Wmax:
                    break

        else:
            # Step 3 — greedy fill free capacity
            zeros = np.where(x == 0)[0]
            zeros = zeros[zeros != 0]
            if len(zeros) > 0:
                order = zeros[np.argsort(-ratio[zeros])]
                for j in order:
                    if total_w + w[j] <= Wmax:
                        x[j] = 1
                        total_w += w[j]

        # Step 4 — NEW: soft speed repair
        x = self._lighten_for_speed(x)

        x[0] = 0
        return x

    # =======================================================
    # Standard Bitflip
    # =======================================================
    def mutateBitflip(self, parent1, parent2, offspring, rate=0.02, **kw):
        n = self.GENOME_LENGTH
        if n == 0:
            return offspring.copy()
        rate = max(0.0, min(float(rate), 1.0))
        child = offspring.copy()
        mask = self._rng.rand(n) < rate
        mask[0] = False
        child[mask] ^= 1
        child[0] = 0
        return child

    # =======================================================
    # Single Bit
    # =======================================================
    def mutateSingleBit(self, parent1, parent2, offspring, **kw):
        n = self.GENOME_LENGTH
        if n <= 1:
            return offspring.copy()
        child = offspring.copy()
        j = self._rng.randint(1, n)
        child[j] ^= 1
        child[0] = 0
        return child

    # =======================================================
    # Mixture of two
    # =======================================================
    # =======================================================
    # Mixture of two
    # =======================================================
    def mutateMixt(self, parent1, parent2, offspring,
                   p_select=None, rate=0.02, **kw):
        if p_select is None:
            p_select = self._configs.get("p_select", [0.7, 0.3])

        p = np.asarray(p_select, dtype=np.float64)
        p /= p.sum()

        choice = self._rng.choice([0, 1], p=p)
        if choice == 0:
            child = self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)
        else:
            child = self.mutateSingleBit(parent1, parent2, offspring, **kw)

        # NEW: soft speed repair
        child = self._lighten_for_speed(child)

        # Always keep depot empty
        if child.shape[0] > 0:
            child[0] = 0
        return child

    # =======================================================
    # Mixture including improved
    # =======================================================
    def mutateMixtExtended(self, parent1, parent2, offspring,
                           p_select=None, rate=0.02, **kw):
        if p_select is None:
            p_select = self._configs.get("p_select", [0.4, 0.3, 0.3])

        p = np.asarray(p_select, dtype=np.float64)
        p /= p.sum()

        choice = self._rng.choice([0, 1, 2], p=p)
        if choice == 0:
            child = self.mutateBitflip(parent1, parent2, offspring, rate=rate, **kw)
        elif choice == 1:
            child = self.mutateSingleBit(parent1, parent2, offspring, **kw)
        else:
            child = self.mutateImproved(parent1, parent2, offspring, rate=rate, **kw)

        # NEW: soft speed repair
        child = self._lighten_for_speed(child)

        # Always keep depot empty
        if child.shape[0] > 0:
            child[0] = 0
        return child

