#!/usr/bin/python
import numpy as np


class FastKPRepair:
    """
    Very cheap knapsack repair used after KP crossover/mutation.

    Given a binary kp vector, drops the worst value/weight items until
    total weight <= Wmax.

    Assumes:
      - kp is a 1D array of 0/1 for items
      - w, v are aligned with kp (same length)
    """

    def __init__(self, w, v, Wmax):
        self.w = np.asarray(w, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.Wmax = float(Wmax)
        self._eps = 1e-12

    def __call__(self, kp):
        # Ensure we work on our own copy
        kp = np.array(kp, copy=True, dtype=np.int32)
        w = self.w

        total_w = float(np.dot(kp, w))
        # Already feasible
        if total_w <= self.Wmax + 1e-9:
            return kp

        # indices of picked items
        idx = np.where(kp == 1)[0]
        if idx.size == 0:
            return kp

        # sort picked items by increasing v/w (worst profit/weight first)
        ratio = self.v[idx] / (self.w[idx] + self._eps)
        order = idx[np.argsort(ratio)]  # worst first

        for i in order:
            kp[i] = 0
            total_w -= w[i]
            if total_w <= self.Wmax + 1e-9:
                break

        # still infeasible? hard cut everything until feasible
        if total_w > self.Wmax + 1e-9:
            left = np.where(kp == 1)[0]
            for i in left:
                kp[i] = 0
                total_w -= w[i]
                if total_w <= self.Wmax + 1e-9:
                    break

        return kp
