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
        self.fitness = None

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

        # ---------------------------------------------------------
        # 0) Optional cooldown: prevents repeated firing
        # ---------------------------------------------------------
        if getattr(self, "_cooldown", 0) > 0:
            self._cooldown -= 1
            return

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

        # ---------------------------------------------------------
        # Access to fitness + alpha
        # ---------------------------------------------------------
        fitness_obj = getattr(self, "fitness", None)

        if fitness_obj is None:
            fit_cfg = {}
            base_alpha = 2.0
        else:
            fit_cfg = getattr(fitness_obj, "_TTPFitness__configs", {})
            base_alpha = fit_cfg.get("alpha", 2.0)

        current_alpha = fit_cfg.get("alpha", base_alpha)

        # ---------------------------------------------------------
        # NO PLATEAU → relax alpha + mutation
        # ---------------------------------------------------------
        if not plateau:

            # Debug
            print(f"[STRESS] No plateau | best_scalar={best_scalar:.4f}, alpha={current_alpha:.4f}")

            # Restore mutation rate
            if self.mutation_boost and self._last_mutation_rate is not None:
                print(f"   • Restoring MUTATION_RATE → {self._last_mutation_rate:.4f}")
                self.setParameters(MUTATION_RATE=self._last_mutation_rate)

            # Relax alpha towards baseline
            if self.dynamic_alpha and fitness_obj is not None:
                if current_alpha > base_alpha:
                    new_alpha = max(base_alpha, current_alpha - 0.05)
                    print(f"   • alpha relaxed: {current_alpha} → {new_alpha}")
                    fit_cfg["alpha"] = new_alpha

            return

        # ---------------------------------------------------------
        # PLATEAU → stress actions
        # ---------------------------------------------------------
        print("\n[STRESS] Plateau detected in StresTTP")
        print(f"   • plateau_window = {self.plateau_window}")
        print(f"   • window values  = {window_vals}")

        # ---------------------------------------------------------
        # Mutation boost
        # ---------------------------------------------------------
        if self.mutation_boost:
            if self._last_mutation_rate is None:
                self._last_mutation_rate = self.MUTATION_RATE

            boost_factor = 2.0
            boosted = min(0.06, self.MUTATION_RATE * boost_factor)

            print(f"   • mutation boosted: {self.MUTATION_RATE:.4f} → {boosted:.4f}")
            self.setParameters(MUTATION_RATE=boosted)

        # ---------------------------------------------------------
        # Dynamic alpha
        # ---------------------------------------------------------
        if self.dynamic_alpha and fitness_obj is not None:
            new_alpha = min(3.5, current_alpha + 0.2)
            print(f"   • alpha increased: {current_alpha} → {new_alpha}")
            fit_cfg["alpha"] = new_alpha

        # ---------------------------------------------------------
        # Shake population
        # ---------------------------------------------------------
        if self.population_shake and (population is not None):
            print(f"   • shaking {int(self.replace_ratio * self.POPULATION_SIZE)} individuals")
            self._shake_population(population)

        # ---------------------------------------------------------
        # Reset history & enable cooldown
        # ---------------------------------------------------------
        self._best_hist.clear()
        self._cooldown = 10   # skip next ~10 gens to avoid spam
        print(f"   • cooldown activated (10 generations)\n")


    # -------------------------------------------------------------
    # Population shake: re-randomize X% of individuals
    # -------------------------------------------------------------
    def _shake_population(self, population):
        """
        Smarter population shake:
        - TSP: apply several random 2-opt style segment reversals
        - KP:  flip ~8% of bits instead of randomizing the whole vector
        """

        try:
            tsp = population["tsp"]
            kp  = population["kp"]
            by_name = True
        except Exception:
            tsp = population[:, 0, :]
            kp  = population[:, 1, :]
            by_name = False

        n_pop = tsp.shape[0]
        n_replace = max(1, int(self.replace_ratio * n_pop))

        idx = self._rng.choice(n_pop, size=n_replace, replace=False)

        def two_opt_swap(route, rng):
            n = route.shape[0]
            i, j = sorted(rng.choice(n, size=2, replace=False))
            if j - i <= 1:
                return
            route[i:j] = route[i:j][::-1]

        for i in idx:
            route = tsp[i]

            # 2-opt noise moves
            n_moves = max(3, int(np.sqrt(route.shape[0]) / 2))
            for _ in range(n_moves):
                two_opt_swap(route, self._rng)

            # SAFE segment rotation (fixed)
            if self._rng.rand() < 0.30:
                L = route.shape[0]

                # ensure we do not pick an 'a' too close to the end
                if L > 10:
                    a = self._rng.randint(0, L - 10)
                    b_low = a + 5
                    b_high = min(a + 35, L)

                    if b_low < b_high:
                        b = self._rng.randint(b_low, b_high)
                        segment = route[a:b]
                        shift = self._rng.randint(1, len(segment))
                        route[a:b] = np.roll(segment, shift)

            # KP shake
            bits = kp[i]
            n_flip = max(5, int(0.08 * bits.shape[0]))
            flip_idx = self._rng.choice(bits.shape[0], n_flip, replace=False)
            bits[flip_idx] ^= 1

        if not by_name:
            population[:, 0, :] = tsp
            population[:, 1, :] = kp

