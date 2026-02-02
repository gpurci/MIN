#!/usr/bin/python
import numpy as np
from extension.kp_repair.cheap_kp_repair import FastKPRepair

class HorizonKPRepair(FastKPRepair):
    """
    Improved KP repair:
      - removes worst-density items first (value/weight)
      - favours adding high-density items late in the TSP tour
      - no extra init arguments beyond base FastKPRepair
    """

    def __init__(self, w, v, Wmax, distance, beta=1.0):
        super().__init__(w, v, Wmax)
        self.distance = distance
        self.beta = beta     # controls how strongly "late" items are favoured

    # -----------------------------------------------------------
    # Remove overweight items (standard + density strategy)
    # -----------------------------------------------------------
    def __call__(self, kp_vec):
        w = self.w
        v = self.v
        Wmax = self.Wmax

        cur_w = np.dot(kp_vec, w)
        if cur_w <= Wmax:
            return kp_vec

        density = v / (w + 1e-9)
        ones = np.where(kp_vec == 1)[0]

        # remove worst density items until valid
        for idx in ones[np.argsort(density[ones])]:
            kp_vec[idx] = 0
            cur_w -= w[idx]
            if cur_w <= Wmax:
                break

        return kp_vec

    # -----------------------------------------------------------
    # Small greedy add of high-density items
    # -----------------------------------------------------------
    def greedy_add_high_density(self, kp_vec, p_add=0.3):
        w = self.w
        v = self.v
        Wmax = self.Wmax

        cur_w = np.dot(kp_vec, w)
        zeros = np.where(kp_vec == 0)[0]

        # high-density sorting
        density = v / (w + 1e-9)
        candidates = zeros[np.argsort(density[zeros])[::-1]]

        for idx in candidates:
            if np.random.rand() > p_add:
                continue
            if cur_w + w[idx] <= Wmax:
                kp_vec[idx] = 1
                cur_w += w[idx]

        return kp_vec
