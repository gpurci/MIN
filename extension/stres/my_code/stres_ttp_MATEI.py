#!/usr/bin/python
import numpy as np

from extension.ga_base import *
from extension.stres.my_code.stres_base import StresBase


class StresTTP(StresBase):
    """
    Plateau-based stress for TTP.

    Behaviour:
      - Tracks a scalar metric from `scores` over last `plateau_window` generations.
      - If that metric is ~constant (np.allclose) over that window:
          → plateau detected → inject diversity.
      - Diversity injection (if enabled):
          → random subset of size `replace_ratio` is strongly randomized (TSP + KP).

      Feature toggles (booleans):
        * dynamic_alpha:
            - on plateau:     alpha += 0.5 (clamped)
            - off plateau:    alpha -= 0.1 down to baseline
        * mutation_boost:
            - on plateau:     temporarily boost MUTATION_RATE
            - off plateau:    restore original MUTATION_RATE
        * population_shake:
            - on plateau:     randomize X% of population (TSP perm + new KP bits)

      NOTE:
        dynamic_alpha only actually changes TTPFitness if this instance has
        a `.fitness` attribute pointing to your fitness object
        (see usage example below).
    """

    def __init__(
        self,
        method="normal",
        subset_size=5,
        plateau_window=15,
        plateau_rtol=5e-3,
        replace_ratio=0.4,
        rng_seed=None,
        dynamic_alpha=True,
        population_shake=True,
        mutation_boost=True,
        **configs,
    ):
        # Register with GA framework
        super().__init__(method, name="StresTTP", **configs)

        # Original param (kept for compatibility / possible use)
        self.subset_size = subset_size

        # Plateau detection settings
        self.plateau_window = plateau_window    # e.g. 10–15
        self.plateau_rtol = plateau_rtol        # how strict "no improvement" is
        self._best_hist = []                    # sliding window for the metric

        # How much of the population to shake (fraction)
        self.replace_ratio = replace_ratio      # e.g. 0.3–0.5

        # Feature toggles
        self.dynamic_alpha = dynamic_alpha
        self.population_shake = population_shake
        self.mutation_boost = mutation_boost

        # RNG
        self._rng = np.random.RandomState(rng_seed)

        # GA parameters (set later via setParameters)
        self.GENOME_LENGTH = None
        self.POPULATION_SIZE = None

        # To remember the "normal" mutation rate when we boost it
        self._last_mutation_rate = None

    # -------------------------------------------------------------
    # GA plumbing
    # -------------------------------------------------------------
    def setParameters(self, **kw):
        """
        Called by GA with things like GENOME_LENGTH, POPULATION_SIZE, etc.
        We forward to GABase and keep a few handy.
        """
        super().setParameters(**kw)

        if "GENOME_LENGTH" in kw:
            self.GENOME_LENGTH = kw["GENOME_LENGTH"]
        if "POPULATION_SIZE" in kw:
            self.POPULATION_SIZE = kw["POPULATION_SIZE"]

    def __repr__(self):
        return (
            f"StresTTP: method '{self._method}'\n"
            f"\tconfigs: '{self._configs}'\n"
        )

    # -------------------------------------------------------------
    # Main operator – GA calls: stres(genoms, scores)
    # -------------------------------------------------------------
    def __call__(self, genoms, scores):
        """
        genoms: population container (Genoms-like)
        scores: dict with aggregated metrics for current generation
                (e.g. {"score": ..., "best_fitness": ..., ...})
        """
        if not isinstance(scores, dict):
            # Defensive: if something weird is passed, do nothing
            return

        # 1) Choose which scalar to track for plateau detection.
        #    Prefer "score" (as in your logs), then "best_fitness", then "fitness".
        metric_val = None
        for k in ("score", "best_fitness", "fitness"):
            if k in scores:
                metric_val = scores[k]
                break

        if metric_val is None:
            # No suitable scalar -> nothing we can do
            return

        try:
            best_scalar = float(metric_val)
        except (TypeError, ValueError):
            # If it’s not a scalar number, bail out safely
            return

        self._do_stress(best_scalar, genoms)

    # -------------------------------------------------------------
    # Core stress logic
    # -------------------------------------------------------------
    def _do_stress(self, best_scalar, population):
        # ---- 1) Update history ----
        self._best_hist.append(best_scalar)
        if len(self._best_hist) > self.plateau_window:
            self._best_hist.pop(0)

        # Not enough history yet → no stress
        if len(self._best_hist) < self.plateau_window:
            return

        window_vals = np.array(self._best_hist, dtype=float)
        plateau = np.allclose(
            window_vals,
            window_vals[-1],
            rtol=self.plateau_rtol,
            atol=0.0,
        )

        # Safe access to fitness config (where alpha lives)
        fitness_obj = getattr(self, "fitness", None)
        if fitness_obj is not None:
            fit_cfg = getattr(fitness_obj, "_Fitness__configs", {})
        else:
            fit_cfg = {}

        base_alpha = fit_cfg.get("alpha", 2.0)

        # ---------------------------------------------------------
        # NO PLATEAU  → relax mutation + alpha back to baseline
        # ---------------------------------------------------------
        if not plateau:
            # Restore mutation rate if we had boosted it
            if self.mutation_boost and self._last_mutation_rate is not None:
                self.setParameters(MUTATION_RATE=self._last_mutation_rate)

            # Slowly decrease alpha towards baseline
            if self.dynamic_alpha and fitness_obj is not None:
                cur_alpha = fit_cfg.get("alpha", base_alpha)
                if cur_alpha > base_alpha:
                    fit_cfg["alpha"] = max(base_alpha, cur_alpha - 0.1)

            return

        # ---------------------------------------------------------
        # PLATEAU  → big but controlled shake
        # ---------------------------------------------------------
        print("[STRESS] Plateau detected in StresTTP. Injecting diversity into population...")

        # --- Mutation boost ---
        if self.mutation_boost:
            if self._last_mutation_rate is None:
                self._last_mutation_rate = self.MUTATION_RATE

            # e.g. 0.01 → up to 0.05–0.1
            boosted = min(0.1, max(self.MUTATION_RATE * 5.0, 0.02))
            self.setParameters(MUTATION_RATE=boosted)

        # --- Dynamic alpha ---
        if self.dynamic_alpha and fitness_obj is not None:
            cur_alpha = fit_cfg.get("alpha", base_alpha)
            new_alpha = min(3.5, cur_alpha + 0.5)  # clamp at 3.5
            fit_cfg["alpha"] = new_alpha
            print(f"   • alpha increased: {cur_alpha} → {new_alpha}")

        # --- Population shake ---
        if self.population_shake and (population is not None):
            self._shake_population(population)

        # After a big shake, reset history so we don’t retrigger immediately
        self._best_hist.clear()

    # -------------------------------------------------------------
    # Population shake: re-randomize X% of individuals
    # -------------------------------------------------------------
    def _shake_population(self, population):
        """
        We don't see per-individual fitness here (GA only passes
        aggregated `scores`), so we can't literally pick "worst Y%".
        Instead, we randomize a random subset of size `replace_ratio`.

        If later you have access to per-individual fitness, you can
        sort by fitness instead of random indices here.
        """
        # Try Genoms-style access by chromosome name first
        try:
            tsp = population["tsp"]   # (n_pop, n_cities)
            kp  = population["kp"]    # (n_pop, n_items)
            by_name = True
        except Exception:
            # Fallback: assume shape (n_pop, 2, L)
            tsp = population[:, 0, :]
            kp  = population[:, 1, :]
            by_name = False

        n_pop = tsp.shape[0]
        n_replace = max(1, int(self.replace_ratio * n_pop))

        # Random indices (since we don't know "worst" individuals)
        idx = self._rng.choice(n_pop, size=n_replace, replace=False)

        for i in idx:
            # Random permutation of cities
            self._rng.shuffle(tsp[i])
            # Fresh random 0/1 vector for items
            kp[i] = self._rng.randint(0, 2, size=kp[i].shape[0])

        # Write back if we used the raw array view
        if not by_name:
            population[:, 0, :] = tsp
            population[:, 1, :] = kp
