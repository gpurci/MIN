import numpy as np
from extension.crossover.crossover_base import *

class CrossoverMateiKP(CrossoverBase):
    """
    Knapsack / binary chromosome crossover operators (Matei combo).

    Methods:
        - 'single_point': single-point crossover.
        - 'two_point'  : two-point crossover.
        - 'uniform'    : uniform crossover.
        - 'mixt'       : mixture between the three above.

    Config:
        - p_select: list/tuple with 3 probabilities [p_sp, p_tp, p_unif] for 'mixt'.
                    If missing or malformed, defaults to [1/3, 1/3, 1/3].
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverMateiKP", **configs)
        self.__fn = self._unpackMethod(
            method,
            single_point=self.crossoverSP,
            two_point=self.crossoverTwoP,
            uniform=self.crossoverUniform,
            weighted=self.crossoverWeighted,
            improved=self.crossoverImproved,
            mixt=self.crossoverMixt,
        )

    def setParameters(self, **kw):
        super().setParameters(**kw)
        
        if "dataset" in kw:
            self.dataset = kw["dataset"]
        
        if "W" in kw:
            self.W = kw["W"]

    def __call__(self, parent1, parent2):
        """
        parent1, parent2: 1D numpy arrays of 0/1 (or general numeric genes).
        """
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverMateiKP:
    metoda: 'single_point'; config None;
    metoda: 'two_point';    config None;
    metoda: 'uniform';      config None;
    metoda: 'mixt';         config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)

    # ------------------------ SINGLE-POINT ----------------------------
    def crossoverSP(self, parent1, parent2, **kw):
        n = self.GENOME_LENGTH
        if n <= 1:
            return parent1.copy()

        # choose cut roughly in the middle region to keep diversity
        sp = np.random.randint(low=n // 4, high=3 * n // 4, size=None)
        offspring = parent1.copy()
        offspring[sp:] = parent2[sp:]
        return offspring

    def crossoverWeighted(self, parent1, parent2, **kw):
        """
        KP weighted crossover:
        - Items with high profit/weight have higher chance of being inherited
        - Helps exploration while keeping good KP structure
        """
        w = self.dataset["item_weight"]
        v = self.dataset["item_profit"]
        ratio = v / (w + 1e-9)

        contrib1 = parent1 * ratio
        contrib2 = parent2 * ratio

        total = contrib1 + contrib2 + 1e-9
        prob = contrib1 / total     # prob of inheriting from p1

        rnd = np.random.rand(self.GENOME_LENGTH)
        offspring = np.where(rnd < prob, parent1, parent2)

        return offspring

    def crossoverImproved(self, parent1, parent2):
        """
        Improved KP crossover:
        1. Randomly choose segments from parents
        2. Repair overweight by removing worst items
        3. Greedy-fill remaining space with best ratio items
        """
        w = self.dataset["item_weight"]
        v = self.dataset["item_profit"]
        Wmax = self.W
        ratio = v / (w + 1e-9)

        # Step 1: partial uniform crossover
        mask = np.random.rand(self.GENOME_LENGTH) < 0.5
        child = np.where(mask, parent1, parent2).copy()

        # Current weight
        total_w = np.sum(child * w)

        # -------------------------------------------------
        # Step 2: If overweight → remove worst ratio items
        # -------------------------------------------------
        if total_w > Wmax:
            order = np.argsort(ratio)    # worst first

            for i in order:
                if child[i] == 1:
                    child[i] = 0
                    total_w -= w[i]
                    if total_w <= Wmax:
                        break

        # -------------------------------------------------
        # Step 3: Fill with best ratio items
        # -------------------------------------------------
        else:
            order = np.argsort(-ratio)   # best first

            for i in order:
                if child[i] == 0 and total_w + w[i] <= Wmax:
                    child[i] = 1
                    total_w += w[i]

        return child

    # ------------------------ TWO-POINT -------------------------------
    def crossoverTwoP(self, parent1, parent2, **kw):
        n = self.GENOME_LENGTH
        if n <= 2:
            return self.crossoverSP(parent1, parent2)

        start, end = np.random.randint(low=1, high=n - 1, size=2)
        if start > end:
            start, end = end, start
        if start == end:
            end = min(start + 1, n - 1)

        offspring = parent1.copy()
        offspring[start:end] = parent2[start:end]
        return offspring

    # ------------------------- UNIFORM --------------------------------
    def crossoverUniform(self, parent1, parent2, **kw):
        n = self.GENOME_LENGTH
        if n == 0:
            return parent1.copy()

        mask = np.random.randint(0, 2, size=n, dtype=bool)
        offspring = parent1.copy()
        offspring[mask] = parent2[mask]
        return offspring


    def crossoverMixt(self, parent1, parent2, p_select=None, **kw):
        """
        Mixture between SP, Two-Point, Uniform, Weighted, Improved crossover.
        """
        if p_select is None:
            p_select = self._configs.get("p_select", [0.2, 0.2, 0.2, 0.2, 0.2])

        # ensure valid probability distribution
        p = np.array(p_select, dtype=float)
        p = p / p.sum()

        choice = np.random.choice([0, 1, 2, 3, 4], p=p)

        if choice == 0:
            return self.crossoverSP(parent1, parent2)
        elif choice == 1:
            return self.crossoverTwoP(parent1, parent2)
        elif choice == 2:
            return self.crossoverUniform(parent1, parent2)
        elif choice == 3:
            return self.crossoverWeighted(parent1, parent2)
        else:
            return self.crossoverImproved(parent1, parent2)

