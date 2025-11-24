import numpy as np
from extension.mutate.my_code.mutate_base import *

# =======================================================================
#   KP / BINARY MUTATION OPERATORS (Matei)
# =======================================================================

class MutateMateiKP(MutateBase):
    """
    Knapsack / binary chromosome mutation operators.

    Methods:
        - 'bitflip'         : flip bits with probability 'rate'
        - 'single_bit'      : flip exactly one bit
        - 'improved'        : bitflip + repair + greedy fill
        - 'mixt'            : mix (bitflip + single_bit)
        - 'mixt_extended'   : mix (bitflip + single_bit + improved)

    Config:
        - rate      : probability in bitflip (default 0.02)
        - p_select  : probabilities for mixtures
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

    # =======================================================
    # Allow dataset to be attached by GA
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
        return self.__fn(parent1, parent2, offspring, **self._configs)

    # =======================================================
    # Improved KP mutation
    # =======================================================
    def mutateImproved(self, parent1, parent2, offspring, rate=0.02, **kw):
        """
        1. Flip a few bits
        2. If overweight → remove worst items
        3. If underweight → add best ratio items
        """
        x = offspring.copy()
        n = self.GENOME_LENGTH

        # Load dataset (TTP items)
        w = self.dataset["item_weight"]
        v = self.dataset["item_profit"]
        Wmax = self.W
        # Step 1 — controlled random flips
        k = max(1, int(n * rate))
        idx = np.random.choice(n, k, replace=False)
        x[idx] ^= 1

        # Evaluate weight
        total_w = np.sum(x * w)
        ratio = v / (w + 1e-9)

        # Step 2 — repair overweight
        if total_w > Wmax:
            order = np.argsort(ratio)        # worst first
            for i in order:
                if x[i] == 1:
                    x[i] = 0
                    total_w -= w[i]
                    if total_w <= Wmax:
                        break
        else:
            # Step 3 — greedy fill
            order = np.argsort(-ratio)       # best first
            for i in order:
                if x[i] == 0 and total_w + w[i] <= Wmax:
                    x[i] = 1
                    total_w += w[i]

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
        mask = np.random.rand(n) < rate
        child[mask] ^= 1
        return child

    # =======================================================
    # Single bit flip
    # =======================================================
    def mutateSingleBit(self, parent1, parent2, offspring, **kw):
        n = self.GENOME_LENGTH
        if n == 0:
            return offspring.copy()

        child = offspring.copy()
        i = np.random.randint(0, n)
        child[i] ^= 1
        return child

    # =======================================================
    # Mixture: bitflip + single bit
    # =======================================================
    def mutateMixt(self, parent1, parent2, offspring, p_select=None, rate=0.02, **kw):
        if p_select is None:
            p_select = self._configs.get("p_select", [0.7, 0.3])

        p = np.array(p_select, dtype=float)
        p /= p.sum()

        choice = np.random.choice([0, 1], p=p)
        if choice == 0:
            return self.mutateBitflip(parent1, parent2, offspring, rate=rate)
        else:
            return self.mutateSingleBit(parent1, parent2, offspring)

    # =======================================================
    # Mixture: bitflip + single bit + improved
    # =======================================================
    def mutateMixtExtended(self, parent1, parent2, offspring,
                           p_select=None, rate=0.02, **kw):

        if p_select is None:
            p_select = self._configs.get("p_select", [0.4, 0.3, 0.3])

        p = np.array(p_select, dtype=float)
        p /= p.sum()

        choice = np.random.choice([0, 1, 2], p=p)

        if choice == 0:
            return self.mutateBitflip(parent1, parent2, offspring, rate=rate)
        elif choice == 1:
            return self.mutateSingleBit(parent1, parent2, offspring)
        else:
            return self.mutateImproved(parent1, parent2, offspring, rate=rate)
