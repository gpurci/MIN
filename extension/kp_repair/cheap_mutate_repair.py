#!/usr/bin/python
import numpy as np

from GeneticAlgorithmManager.my_code.root_GA import RootGA
from extension.kp_repair.destroy_repair_kp import DestroyRepairKP


class MutateKPWithRepairAndStress(RootGA):
    """
    Wrapper over a base KP mutator that:
      1) applies base_mutator
      2) repairs overweight (external repair, then worst-density removal)
      3) optionally applies stress destroy/repair when stress_mode=True
      4) optional mild greedy density add
      5) ALWAYS enforces kp[0] = 0

    NOTE:
    - When used with StresTTP, stress is toggled via set_stress_mode()/stress_mode.
    - When used alone (no stress), stress_mode stays False and it behaves like
      "mutate + repair + mild greedy add".
    """

    def __init__(
        self,
        base_mutator,
        W,
        profits,
        weights,
        stress_destroy_frac=0.6,
        stress_destroy_worst_frac=0.3,
        allow_add=True,
        repair=None,
        use_density_mutation=True,
        n_try_density=3,
        rng=None,
        **configs,
    ):
        super().__init__()

        self.base_mutator = base_mutator
        self.W = float(W)

        self.v = np.asarray(profits, dtype=np.float64)
        self.w = np.asarray(weights, dtype=np.float64)

        if self.v.shape != self.w.shape:
            raise ValueError("profits and weights must have same shape")

        # safe density
        eps = 1e-9
        density = self.v / (self.w + eps)
        density[self.w <= 0] = 0.0
        self.density = density

        self.repair = repair
        self.use_density_mutation = bool(use_density_mutation)
        self.n_try_density = int(n_try_density)

        self._rng = rng if rng is not None else np.random.RandomState()
        self._configs = dict(configs)

        self.destroy_repair = DestroyRepairKP(
            W=self.W,
            profits=self.v,
            weights=self.w,
            frac_drop_random=stress_destroy_frac,
            frac_drop_worst=stress_destroy_worst_frac,
            allow_add=allow_add,
            rng=self._rng,
        )

        # This is what StresTTP toggles
        self.stress_mode = False

    # -------------------------------------------------------
    # GA integration
    # -------------------------------------------------------
    def __str__(self):
        return f"MutateKPWithRepairAndStress(W={self.W}, stress_mode={self.stress_mode})"

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if hasattr(self.base_mutator, "setParameters"):
            self.base_mutator.setParameters(**kw)
        if hasattr(self.destroy_repair, "setParameters"):
            self.destroy_repair.setParameters(**kw)

    def set_stress_mode(self, on):
        """Preferred API for StresTTP to toggle stress behaviour."""
        self.stress_mode = bool(on)

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------
    def _repair_overweight(self, kp):
        """Fallback repair: remove lowest-density items until under capacity."""
        x = kp.copy()
        w = self.w
        ratio = self.density

        cur_w = float(np.dot(x, w))
        if cur_w <= self.W:
            return x

        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]

        while cur_w > self.W and ones.size > 0:
            j = ones[np.argmin(ratio[ones])]
            x[j] = 0
            cur_w -= w[j]
            ones = np.where(x == 1)[0]
            ones = ones[ones != 0]

        if x.shape[0] > 0:
            x[0] = 0
        return x

    def _density_mutation(self, kp, n_try=1):
        """
        Mild greedy add: try to add high-density items while remaining under W.
        Only used when stress_mode == False (normal GA evolution).
        """
        x = kp.copy()
        w = self.w
        ratio = self.density
        cur_w = float(np.dot(x, w))

        if cur_w >= self.W:
            return x

        for _ in range(max(1, int(n_try))):
            zeros = np.where(x == 0)[0]
            zeros = zeros[zeros != 0]

            if zeros.size == 0:
                break

            j = zeros[np.argmax(ratio[zeros])]
            if cur_w + w[j] <= self.W:
                x[j] = 1
                cur_w += w[j]
            else:
                break

        if x.shape[0] > 0:
            x[0] = 0
        return x

    # -------------------------------------------------------
    # MAIN CALL (GA interface)
    # -------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **call_configs):
        """
        parent1, parent2, offspring: KP chromosomes (0/1 vectors)
        Returns: repaired / stressed child chromosome (1D np.int8)
        """
        cfg = dict(self._configs)
        cfg.update(call_configs)

        # 1) base mutation
        child = self.base_mutator(parent1, parent2, offspring)
        child = np.asarray(child, dtype=np.int8).copy()

        if child.ndim != 1:
            child = child.ravel()

        if child.shape[0] > 0:
            child[0] = 0

        # 2) Try external repair (FastKPRepair / MultiStrategyKPRepair)
        cur_w = float(np.dot(child, self.w))
        if cur_w > self.W and self.repair is not None:
            repaired = self.repair(child)
            child = np.asarray(repaired, dtype=np.int8).copy()
            if child.shape[0] > 0:
                child[0] = 0
            cur_w = float(np.dot(child, self.w))

        # 3) fallback worst-density repair
        if cur_w > self.W:
            child = self._repair_overweight(child)
            cur_w = float(np.dot(child, self.w))

        # 4) stress-mode destroy/repair
        if self.stress_mode:
            child = self.destroy_repair(parent1, parent2, child)
            if child.shape[0] > 0:
                child[0] = 0

            cur_w = float(np.dot(child, self.w))
            if cur_w > self.W:
                child = self._repair_overweight(child)

        # 5) mild density mutation (only when NOT in stress mode)
        if self.use_density_mutation and not self.stress_mode:
            child = self._density_mutation(child, n_try=self.n_try_density)

        if child.shape[0] > 0:
            child[0] = 0

        return child

class MutateKPWithRepair(MutateKPWithRepairAndStress):
    """
    Variant of MutateKPWithRepairAndStress that never uses stress_mode.

    Use this when you want simple mutate+repair (no destroy/repair stress),
    but with the same interface as the stressed version.
    """
    def __init__(self, *args, **kwargs):
        # keep density mutation unless the caller disables it explicitly
        kwargs.setdefault("use_density_mutation", True)
        super().__init__(*args, **kwargs)
        # make sure stress_mode is OFF
        self.stress_mode = False
