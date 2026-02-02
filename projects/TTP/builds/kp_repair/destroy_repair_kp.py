#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class DestroyRepairKP(RootGA):
    """
    Aggressive destroy & repair operator for KP bitstrings.

    Used from MutateKPWithRepairAndStress in stress mode.

    Steps:
      1) Randomly drop a fraction of taken items
      2) Drop a fraction of worst-density items
      3) If still overweight, keep dropping worst-density
      4) Greedily re-add best-density items while respecting capacity
      5) ALWAYS enforce kp[0] = 0 (no item at depot)
    """

    def __init__(
        self,
        W,
        profits,
        weights,
        frac_drop_random=0.5,
        frac_drop_worst=0.25,
        allow_add=True,
        add_limit=None,
        rng=None,
        **configs,
    ):
        super().__init__()
        self.W = float(W)

        self.v = np.asarray(profits, dtype=np.float64)
        self.w = np.asarray(weights, dtype=np.float64)

        if self.v.shape != self.w.shape:
            raise ValueError("profits and weights must have same shape")

        if not (0.0 <= frac_drop_random <= 1.0):
            raise ValueError("frac_drop_random must be in [0,1]")
        if not (0.0 <= frac_drop_worst <= 1.0):
            raise ValueError("frac_drop_worst must be in [0,1]")

        self.frac_drop_random = float(frac_drop_random)
        self.frac_drop_worst = float(frac_drop_worst)
        self.allow_add = bool(allow_add)
        self.add_limit = add_limit
        self._configs = dict(configs)

        self._rng = rng if rng is not None else np.random.RandomState()

        # robust density
        eps = 1e-9
        density = self.v / (self.w + eps)
        density[self.w <= 0] = 0.0
        self.density = density

    def __str__(self):
        return (
            f"DestroyRepairKP(W={self.W}, "
            f"frac_drop_random={self.frac_drop_random}, "
            f"frac_drop_worst={self.frac_drop_worst}, "
            f"allow_add={self.allow_add})"
        )

    def help(self):
        msg = [
            "DestroyRepairKP:",
            "  • random drop of some taken items",
            "  • drop some worst-density items",
            "  • enforce feasibility (drop worst-density until w <= W)",
            "  • greedy re-add best-density items",
            "  • ALWAYS kp[0] = 0 (depot empty)",
        ]
        print("\n".join(msg))

    def setParameters(self, **kw):
        super().setParameters(**kw)

    # -------------------------------------------------------
    # MAIN CALL
    # -------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **call_configs):
        """
        offspring: 1-D KP bitstring (np array of 0/1).
        """
        x = np.asarray(offspring, dtype=np.int8).copy()
        if x.ndim != 1:
            x = x.ravel()

        n = x.shape[0]
        if n == 0:
            return x

        # depot is always empty
        x[0] = 0

        Wmax = self.W
        w = self.w
        ratio = self.density
        rng = self._rng

        cur_w = float(np.dot(x, w))

        # 1) Random drop of some taken items
        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]  # exclude depot
        if ones.size > 0 and self.frac_drop_random > 0.0:
            k_drop = int(round(self.frac_drop_random * ones.size))
            k_drop = max(0, min(k_drop, ones.size))
            if k_drop > 0:
                idx_drop = rng.choice(ones, size=k_drop, replace=False)
                x[idx_drop] = 0
                cur_w = float(np.dot(x, w))

        # 2) Drop some worst-density items
        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]
        if ones.size > 0 and self.frac_drop_worst > 0.0:
            k_drop = int(round(self.frac_drop_worst * ones.size))
            k_drop = max(0, min(k_drop, ones.size))
            if k_drop > 0:
                order = np.argsort(ratio[ones])  # worst first
                idx_drop = ones[order[:k_drop]]
                x[idx_drop] = 0
                cur_w = float(np.dot(x, w))

        # 3) Ensure feasibility by dropping worst-density until w <= W
        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]
        while ones.size > 0 and cur_w > Wmax:
            j = ones[np.argmin(ratio[ones])]
            x[j] = 0
            cur_w -= w[j]
            ones = np.where(x == 1)[0]
            ones = ones[ones != 0]

        # 4) Greedy re-add best-density items
        if self.allow_add and cur_w < Wmax:
            zeros = np.where(x == 0)[0]
            zeros = zeros[zeros != 0]
            if zeros.size > 0:
                order = zeros[np.argsort(-ratio[zeros])]  # best first
                added = 0
                for j in order:
                    if cur_w + w[j] <= Wmax:
                        x[j] = 1
                        cur_w += w[j]
                        added += 1
                        if self.add_limit is not None and added >= self.add_limit:
                            break

        # final invariant
        x[0] = 0
        return x
