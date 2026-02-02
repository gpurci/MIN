#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class MultiStrategyKPRepair(RootGA):
    """
    Multi-strategy KP repair for TTP.

    - Used after KP crossover / mutation when the knapsack is overweight.
    - Combines:
        * FAST     : simple worst-density removal until feasible.
        * HORIZON  : fast repair + gentle greedy add (v/w) to refill.
        * AGGRESSIVE: a bit more random drop + refill.

    - It is also "stress-aware":
        * mode = "normal"  → uses p_modes_normal (e.g. (0.4, 0.4, 0.2)).
        * mode = "stress"  → uses p_modes_stress (by default FAST-only).

      Call set_mode("normal") / set_mode("stress") from StresTTP
      (or anywhere else) to coordinate with stress.
    """

    def __init__(
        self,
        w,
        v,
        Wmax,
        distance=None,
        beta=1.0,
        p_modes=(0.4, 0.4, 0.2),
        min_rel_over_for_multi=0.05,
        rng=None,
        **configs,
    ):
        """
        Parameters
        ----------
        w : 1D array
            Item weights.
        v : 1D array
            Item profits.
        Wmax : float
            Knapsack capacity.
        distance : array-like, optional
            TSP distance matrix (not heavily used here, but kept for API).
        beta : float
            Trade-off parameter (reserved for more TTP-aware scoring, if needed).
        p_modes : tuple of 3 floats
            Probabilities for (FAST, HORIZON, AGGRESSIVE) in **normal** mode.
        min_rel_over_for_multi : float
            If relative overload < this threshold, we do only FAST repair.
        rng : np.random.RandomState, optional
            RNG for stochastic parts.
        configs : dict
            Additional configs (stored, not used directly).
        """
        super().__init__()
        self.w = np.asarray(w, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        if self.w.shape != self.v.shape:
            raise ValueError("w and v must have the same shape")

        self.Wmax = float(Wmax)
        self.distance = distance  # kept for API, not mandatory here
        self.beta = float(beta)
        self._configs = dict(configs)

        # basic density v / w (like in your KP mutator)
        eps = 1e-9
        density = self.v / (self.w + eps)
        density[self.w <= 0] = 0.0
        self.density = density

        self.min_rel_over_for_multi = float(min_rel_over_for_multi)

        self._rng = rng if rng is not None else np.random.RandomState()

        # --- mode handling (normal vs stress) ---
        p_modes = np.asarray(p_modes, dtype=np.float64)
        if p_modes.shape != (3,):
            raise ValueError("p_modes must be a length-3 tuple/list for (FAST, HORIZON, AGGRESSIVE)")

        if p_modes.sum() <= 0:
            p_modes = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        p_modes /= p_modes.sum()

        # probabilities for normal mode
        self.p_modes_normal = p_modes.copy()
        # by default, stress mode = FAST-only (you can tweak this if you want)
        self.p_modes_stress = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        self.mode = "normal"
        self._update_active_p_modes()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_active_p_modes(self):
        if self.mode == "stress":
            self.p_modes = self.p_modes_stress.copy()
        else:
            self.p_modes = self.p_modes_normal.copy()

    def set_mode(self, mode: str):
        """
        Set repair mode: "normal" or "stress".

        Called from StresTTP so that multi-repair and KP stress
        are coordinated.
        """
        mode = str(mode).lower()
        if mode not in ("normal", "stress"):
            raise ValueError("MultiStrategyKPRepair.set_mode: mode must be 'normal' or 'stress'")
        self.mode = mode
        self._update_active_p_modes()

    # classic worst-density drop until feasible
    def _fast_repair(self, x):
        x = x.copy()
        w = self.w
        ratio = self.density

        cur_w = float(np.dot(x, w))
        if cur_w <= self.Wmax + 1e-9:
            return x

        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]  # never pick item 0

        while cur_w > self.Wmax + 1e-9 and ones.size > 0:
            # remove lowest-density item
            j = ones[np.argmin(ratio[ones])]
            x[j] = 0
            cur_w -= w[j]
            ones = np.where(x == 1)[0]
            ones = ones[ones != 0]

        if x.shape[0] > 0:
            x[0] = 0
        return x

    def _greedy_add(self, x, max_steps=10):
        """Try to add good-density items without exceeding capacity."""
        x = x.copy()
        w = self.w
        ratio = self.density
        cur_w = float(np.dot(x, w))

        for _ in range(max_steps):
            zeros = np.where(x == 0)[0]
            zeros = zeros[zeros != 0]
            if zeros.size == 0:
                break

            j = zeros[np.argmax(ratio[zeros])]
            if cur_w + w[j] <= self.Wmax + 1e-9:
                x[j] = 1
                cur_w += w[j]
            else:
                break

        if x.shape[0] > 0:
            x[0] = 0
        return x

    def _horizon_repair(self, x):
        """
        Gentle: fast repair + small greedy refill.
        """
        x = self._fast_repair(x)
        x = self._greedy_add(x, max_steps=5)
        return x

    def _aggressive_repair(self, x):
        """
        Slightly more exploratory repair:
        - random drop of a few items, then fast repair + bigger refill.
        """
        x = x.copy()
        ones = np.where(x == 1)[0]
        ones = ones[ones != 0]

        if ones.size > 0:
            k_drop = max(1, int(0.05 * ones.size))
            j_drop = self._rng.choice(ones, size=k_drop, replace=False)
            x[j_drop] = 0

        x = self._fast_repair(x)
        x = self._greedy_add(x, max_steps=15)
        return x

    # ------------------------------------------------------------------
    # MAIN CALL
    # ------------------------------------------------------------------
    def __call__(self, kp):
        """
        kp : 1D 0/1 vector.

        Returns a FEASIBLE 0/1 vector (w^T x <= Wmax) with some
        combination of fast / horizon / aggressive repair applied.
        """
        x = np.asarray(kp, dtype=np.int8).copy()
        if x.ndim != 1:
            x = x.ravel()

        if x.shape[0] > 0:
            x[0] = 0

        cur_w = float(np.dot(x, self.w))
        if cur_w <= self.Wmax + 1e-9:
            # already feasible → just make sure item 0 is off
            return x

        # how badly overweight are we?
        rel_over = (cur_w - self.Wmax) / (self.Wmax + 1e-9)
        if rel_over < self.min_rel_over_for_multi:
            # near-feasible → use only fast repair
            x = self._fast_repair(x)
            return x

        # choose a mode according to current mode's probabilities
        mode_idx = int(self._rng.choice(3, p=self.p_modes))
        if mode_idx == 0:
            x = self._fast_repair(x)
        elif mode_idx == 1:
            x = self._horizon_repair(x)
        else:
            x = self._aggressive_repair(x)

        # final safety: if still overweight (numerically), fall back to fast
        final_w = float(np.dot(x, self.w))
        if final_w > self.Wmax + 1e-9:
            x = self._fast_repair(x)

        x = np.asarray(x, dtype=np.int8)
        if x.shape[0] > 0:
            x[0] = 0
        return x
