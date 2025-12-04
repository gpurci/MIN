#!/usr/bin/python

import numpy as np
from extension.crossover.crossover_base import *

class CrossoverMateiTSP(CrossoverBase):
    """
    TSP / permutation crossover operators (Matei combo).

    Methods:
        - 'ox'   : classic Order Crossover (OX).
        - 'cycle': Cycle Crossover (CX).
        - 'mixt' : mixture between 'ox' and 'cycle', controlled by p_select.

    Config (passed via **configs in __init__):
        - p_select: list/tuple with 2 probabilities [p_ox, p_cycle] for 'mixt'.
                    If missing or malformed, defaults to [0.7, 0.3].
    """

    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverMateiTSP", **configs)
        self.__fn = self._unpackMethod(
            method,
            ox=self.crossoverOX,
            cycle=self.crossoverCycle,
            mixt=self.crossoverMixt,
        )

    def __call__(self, parent1, parent2):
        """
        parent1, parent2: 1D numpy arrays encoding a permutation of cities.
        """
        return self.__fn(parent1, parent2, **self._configs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def help(self):
        info = """CrossoverMateiTSP:
    metoda: 'ox';    config None;
    metoda: 'cycle'; config None;
    metoda: 'mixt';  config -> "p_select":[0.7, 0.3], ;\n"""
        print(info)

    # ----------------------------- OX ---------------------------------
    def crossoverOX(self, parent1, parent2, **kw):
        """
        Classic Order Crossover (OX) - returns a single offspring.
        Keeps a slice from parent1, fills remaining positions from parent2
        preserving order and permutation constraint.
        """
        n = self.GENOME_LENGTH
        if n <= 1:
            return parent1.copy()

        # choose two cut points
        cut1, cut2 = np.random.randint(0, n, size=2)
        if cut1 > cut2:
            cut1, cut2 = cut2, cut1
        elif cut1 == cut2:
            cut2 = (cut1 + 1) % n
            if cut1 > cut2:
                cut1, cut2 = cut2, cut1

        offspring = np.full(n, -1, dtype=parent1.dtype)

        # copy segment from parent1
        offspring[cut1:cut2 + 1] = parent1[cut1:cut2 + 1]
        used = set(parent1[cut1:cut2 + 1])

        # fill the rest from parent2 in order, skipping used genes
        pos = (cut2 + 1) % n
        # iterate parent2 starting from cut2+1, wrapping around
        ordered_p2 = np.concatenate((parent2[cut2 + 1:], parent2[:cut2 + 1]))
        for gene in ordered_p2:
            if gene in used:
                continue
            # find next empty position
            while offspring[pos] != -1:
                pos = (pos + 1) % n
            offspring[pos] = gene
            used.add(int(gene))

        return offspring

    # ---------------------------- CYCLE -------------------------------
    def crossoverCycle(self, parent1, parent2, **kw):
        """
        Cycle crossover (CX).
        Builds cycles between parent1 and parent2; odd cycles come from
        parent1, even cycles from parent2.
        """
        n = self.GENOME_LENGTH
        if n <= 1:
            return parent1.copy()

        offspring = np.empty_like(parent1)
        assigned = np.zeros(n, dtype=bool)

        take_from_p1 = True

        while not np.all(assigned):
            # start new cycle from first unassigned index
            start_idx = int(np.where(~assigned)[0][0])
            idx = start_idx
            cycle_indices = []

            while True:
                cycle_indices.append(idx)
                assigned[idx] = True

                # value from parent2 at current index
                val = parent2[idx]
                # next index is where parent1 has this value
                next_idx_arr = np.where(parent1 == val)[0]
                if next_idx_arr.size == 0:
                    break  # safety: shouldn't happen for valid permutations
                idx = int(next_idx_arr[0])

                if assigned[idx]:
                    break

            cycle_indices = np.array(cycle_indices, dtype=int)
            if take_from_p1:
                offspring[cycle_indices] = parent1[cycle_indices]
            else:
                offspring[cycle_indices] = parent2[cycle_indices]

            take_from_p1 = not take_from_p1

        return offspring

    # --------------------------- MIXTURE ------------------------------
    def crossoverMixt(self, parent1, parent2, p_select=None, **kw):
        """
        Mixture between OX and Cycle crossover.

        p_select: [p_ox, p_cycle] - probability of each operator.
                  If None or invalid, defaults to [0.7, 0.3].
        """
        if p_select is None:
            p_select = self._configs.get("p_select", [0.7, 0.3])

        try:
            p = np.array(p_select, dtype=float)
            if p.size != 2 or np.any(p < 0):
                raise ValueError
            s = p.sum()
            if s <= 0:
                raise ValueError
            p /= s
        except Exception:
            p = np.array([0.7, 0.3], dtype=float)

        choice = np.random.choice([0, 1], p=p)
        if choice == 0:
            return self.crossoverOX(parent1, parent2)
        else:
            return self.crossoverCycle(parent1, parent2)

