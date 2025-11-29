#!/usr/bin/python
import numpy as np

from extension.ga_base import *
from extension.stres.my_code.stres_base import StresBase


class StresTTP(StresBase):
    """
    Plateau-based stress for TTP (lighter version, no heavy re-inits by default).

    Behaviour (per GA generation):
      * Reads `scores["best_fitness"]` (TTP fitness – lower is better).
      * Detects plateaus: no *significant* improvement for `plateau_window`
        generations. Significance is controlled by `plateau_rtol`.
      * On a plateau:
          - Optionally boosts MUTATION_RATE (capped).
          - Turns on KP "stress mode" (MutateKPWithRepairAndStress).
          - Shakes part of the population by re-mutating some individuals.
      * When a new global best is found:
          - Resets plateau counter.
          - Turns KP stress mode off.
          - Gradually relaxes MUTATION_RATE towards the base value.

    This class is stateful across calls; the GA calls it once per
    generation as:  `stres(genoms, scores)`.
    """

    def __init__(
        self,
        plateau_window=15,
        plateau_rtol=1e-3,
        replace_ratio=0.20,
        mutation_boost=True,
        mutation_boost_factor=1.25,
        mutation_rate_max=0.12,
        dynamic_alpha=True,
        restart_ratio=0.0,
        rng=None,
        **configs,
    ):
        # We don't actually use the method dispatch of GABase here,
        # but we keep the parent initialisation for consistency.
        super().__init__(method="noop", name="StresTTP", **configs)

        # --- hyper-parameters ---
        self.plateau_window        = int(plateau_window)
        self.plateau_rtol          = float(plateau_rtol)
        self.replace_ratio         = float(replace_ratio)
        self.mutation_boost        = bool(mutation_boost)
        self.mutation_boost_factor = float(mutation_boost_factor)
        self.mutation_rate_max     = float(mutation_rate_max)
        self.dynamic_alpha         = bool(dynamic_alpha)
        # kept for backwards compatibility; by default we do *not*
        # perform partial restarts anymore (only shaking)
        self.restart_ratio         = float(restart_ratio)

        # External hooks (wired from builder):
        #   - fitness         : TTPFitness object (OPTIONAL)
        #   - mutate_kp       : MutateKPWithRepairAndStress instance
        #   - mutate_tsp      : MutateMateiTSP instance (OPTIONAL, for shaking)
        #   - init_population : InitPopulationHybrid instance (OPTIONAL)
        self.fitness         = None
        self.mutate_kp       = None
        self.mutate_tsp      = None
        self.init_population = None

        # Internal state
        self._rng = rng if rng is not None else np.random.RandomState()

        self._best_fitness     = np.inf   # best so far (lower is better)
        self._no_improve_count = 0        # plateau counter
        self._generation       = 0        # internal generation counter

        self._base_mutation_rate = None   # set in setParameters()
        self.MUTATION_RATE       = None   # current stress-side view

    # -----------------------------------------------------------
    # Standard GA-style parameter propagation
    # -----------------------------------------------------------
    def setParameters(self, **kw):
        """
        Called by the GA once at the beginning; we use it mainly to
        remember the base MUTATION_RATE and (optionally) POPULATION_SIZE.
        """
        super().setParameters(**kw)

        if "MUTATION_RATE" in kw:
            self._base_mutation_rate = float(kw["MUTATION_RATE"])
            self.MUTATION_RATE       = float(kw["MUTATION_RATE"])
        elif self._base_mutation_rate is None:
            # Fallback default if nothing is provided.
            self._base_mutation_rate = 0.05
            self.MUTATION_RATE       = 0.05

        # Not strictly needed, but may be useful.
        self.POPULATION_SIZE = kw.get("POPULATION_SIZE", kw.get("POP_SIZE", None))

    # -----------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------
    def _extract_best_fitness(self, scores):
        """
        Extract scalar 'best_fitness' from the `scores` dict.

        MetricsTTPMATEI.getScoreTTP exposes scores["best_fitness"] as
        the best (lowest) fitness in the population.
        """
        if scores is None:
            return None
        if "best_fitness" not in scores:
            return None
        try:
            v = np.asarray(scores["best_fitness"]).astype(float)
            # handle scalar or length-1 array
            return float(v.ravel()[0])
        except Exception:
            return None

    def _ssd_score(self, scores, idx):
        """
        Per-individual "cost" used for ranking in _shake_population.

        We use -score[i] so that *smaller* is *better* (as in a loss),
        given that TTP 'score' is larger-is-better.
        """
        if scores is None:
            return 0.0
        s = scores.get("score", None)
        if s is None:
            return 0.0
        s = np.asarray(s)
        if s.ndim == 0:
            return float(-s)
        if idx < 0 or idx >= s.shape[0]:
            return 0.0
        return float(-s[idx])

    def _elite_mask(self, genoms, n):
        """
        Returns a boolean mask of length n marking which indices are elites,
        based on Genoms.getElitePos() (if available).
        """
        elite_idx = []
        if hasattr(genoms, "getElitePos"):
            try:
                elite_idx = list(genoms.getElitePos())
            except Exception:
                elite_idx = []

        mask = np.zeros(n, dtype=bool)
        elite_idx = [i for i in elite_idx if 0 <= i < n]
        mask[elite_idx] = True
        return mask

    # -----------------------------------------------------------
    # Population shaking
    # -----------------------------------------------------------
    def _shake_population(self, genoms, scores, full=False):
        """
        Cheap population shake:
          - keep top n_elite untouched
          - pick some of the worst individuals
          - rebuild them as mutated clones of better individuals

        This does *not* call InitPopulationHybrid, so it is much cheaper
        than a full re-initialisation.
        """
        # underlying numpy arrays
        tsp_pop = genoms["tsp"]
        kp_pop  = genoms["kp"]

        pop_size = tsp_pop.shape[0]
        if pop_size < 10:
            return  # nothing useful to do

        # ---- 1) Rank individuals (smaller "cost" is better) ----
        ssd_scores = np.array([self._ssd_score(scores, i) for i in range(pop_size)])
        idx_sorted = np.argsort(ssd_scores)

        # top 5% or at least 5 elites
        n_elite = max(5, int(0.05 * pop_size))
        n_elite = min(n_elite, pop_size // 2)  # don’t let elites be most of population

        elite_idx = idx_sorted[:n_elite]
        worst_idx = idx_sorted[n_elite:]

        if worst_idx.size == 0:
            return

        # ---- 2) Decide how many to replace ----
        # For “full” shake we can replace more, but still not 100%.
        if full:
            frac = min(0.5, self.replace_ratio * 2.0)   # up to 50%
        else:
            frac = self.replace_ratio                   # e.g. 0.20

        n_replace = max(1, int(frac * worst_idx.size))
        n_replace = min(n_replace, worst_idx.size)

        replace_idx = self._rng.choice(worst_idx, size=n_replace, replace=False)

        # ---- 3) For each replaced individual: clone + strong mutation ----
        for idx in replace_idx:
            # pick donor from elites
            donor = int(self._rng.choice(elite_idx))

            new_tsp = tsp_pop[donor].copy()
            new_kp  = kp_pop[donor].copy()

            # safety: ensure chromosome shapes ok
            new_tsp = np.asarray(new_tsp, dtype=np.int64).ravel()
            new_kp  = np.asarray(new_kp,  dtype=np.int8).ravel()

            # --- TSP “shake”: several mutations on route (if wired) ---
            if self.mutate_tsp is not None:
                for _ in range(3):
                    new_tsp = self.mutate_tsp(new_tsp, new_tsp, new_tsp)

            # --- KP “shake”: rely on current stress_mode & repair ---
            if self.mutate_kp is not None:
                new_kp = self.mutate_kp(new_kp, new_kp, new_kp)

            # enforce kp[0] = 0 if needed
            if new_kp.shape[0] > 0:
                new_kp[0] = 0

            tsp_pop[idx] = new_tsp
            kp_pop[idx]  = new_kp

    # -----------------------------------------------------------
    # MUTATION_RATE handling
    # -----------------------------------------------------------
    def _boost_mutation_rate(self):
        """
        Increase MUTATION_RATE (both here and inside mutate_kp) using
        `mutation_boost_factor`, but never above `mutation_rate_max`.
        """
        if not self.mutation_boost:
            return

        if self.MUTATION_RATE is None:
            return

        new_rate = self.MUTATION_RATE * self.mutation_boost_factor
        new_rate = float(min(self.mutation_rate_max, new_rate))

        if new_rate <= self.MUTATION_RATE + 1e-12:
            return

        self.MUTATION_RATE = new_rate

        # Try to propagate to KP mutator if it has a setParameters() or attribute.
        try:
            if self.mutate_kp is not None and hasattr(self.mutate_kp, "setParameters"):
                self.mutate_kp.setParameters(MUTATION_RATE=new_rate)
            elif self.mutate_kp is not None and hasattr(self.mutate_kp, "MUTATION_RATE"):
                self.mutate_kp.MUTATION_RATE = new_rate
        except Exception:
            pass

    def _relax_mutation_rate(self):
        """
        Slowly bring MUTATION_RATE back towards the base value when
        we escape a plateau.
        """
        if self._base_mutation_rate is None or self.MUTATION_RATE is None:
            return

        if self.MUTATION_RATE <= self._base_mutation_rate:
            self.MUTATION_RATE = self._base_mutation_rate
            return

        # simple geometric decay back towards the base
        new_rate = 0.5 * (self.MUTATION_RATE + self._base_mutation_rate)
        self.MUTATION_RATE = max(self._base_mutation_rate, new_rate)

        try:
            if self.mutate_kp is not None and hasattr(self.mutate_kp, "setParameters"):
                self.mutate_kp.setParameters(MUTATION_RATE=self.MUTATION_RATE)
            elif self.mutate_kp is not None and hasattr(self.mutate_kp, "MUTATION_RATE"):
                self.mutate_kp.MUTATION_RATE = self.MUTATION_RATE
        except Exception:
            pass

    # -----------------------------------------------------------
    # KP stress + multi-repair coordination
    # -----------------------------------------------------------
    def _toggle_kp_stress(self, on):
        """
        Turn KP stress on/off AND coordinate the multi-repair behaviour.

        - mutate_kp.stress_mode (or set_stress_mode) controls the
          destroy+repair in MutateKPWithRepairAndStress.
        - If mutate_kp.repair has set_mode("normal"/"stress"), we call it
          so that MultiStrategyKPRepair (or others) are also aware of
          the stress state.
        """
        on = bool(on)
        mk = self.mutate_kp
        if mk is None:
            return

        # 1) toggle stress flag on the KP mutator
        try:
            if hasattr(mk, "set_stress_mode"):
                mk.set_stress_mode(on)
            elif hasattr(mk, "stress_mode"):
                mk.stress_mode = on
        except Exception:
            pass

        # 2) if there is an attached repair object with a set_mode API,
        #    let it know whether we are in stress or not
        try:
            repair = getattr(mk, "repair", None)
            if repair is not None and hasattr(repair, "set_mode"):
                if on:
                    repair.set_mode("stress")
                else:
                    repair.set_mode("normal")
        except Exception:
            pass

    # -----------------------------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------------------------
    def __call__(self, genoms, scores):
        """
        Main stress hook - called once per generation by the GA.

        Parameters
        ----------
        genoms : Genoms
            Population container.
        scores : dict
            MetricsTTPMATEI.getScoreTTP(...) result. Must contain
            "best_fitness" for plateau detection and "score" for
            per-individual ranking.
        """
        if genoms is None or scores is None:
            return

        self._generation += 1

        cur_best = self._extract_best_fitness(scores)
        if cur_best is None or not np.isfinite(cur_best):
            # If we can't read a sensible metric, do nothing.
            return

        # ---------------------------------------------------
        # Detect significant improvement (escape from plateau)
        # ---------------------------------------------------
        if np.isfinite(self._best_fitness):
            # improvement must be at least plateau_rtol fraction
            threshold = self._best_fitness * (1.0 - self.plateau_rtol)
        else:
            threshold = np.inf  # first call

        if cur_best < threshold:
            # New global best ⇒ reset plateau, relax stress.
            self._best_fitness     = cur_best
            self._no_improve_count = 0

            # turn OFF KP stress mode (and normalise repair) if available
            self._toggle_kp_stress(False)

            # gently relax MUTATION_RATE towards base
            self._relax_mutation_rate()
            return

        # ---------------------------------------------------
        # No significant improvement ⇒ increase plateau counter
        # ---------------------------------------------------
        if not np.isfinite(self._best_fitness):
            # This is the very first recorded best.
            self._best_fitness = cur_best
            self._no_improve_count = 0
            return

        self._no_improve_count += 1

        # below threshold ⇒ no stress yet
        if self._no_improve_count < self.plateau_window:
            return

        # We are in plateau territory ⇒ optional mutation boost
        self._boost_mutation_rate()

        # Turn ON KP stress mode (including multi-repair, if supported)
        self._toggle_kp_stress(True)

        # ---------------------------------------------------
        # Mild vs stronger plateau:
        #   [plateau_window, 2*plateau_window)  => mild shaking
        #   [2*plateau_window, ∞)              => stronger shaking
        # ---------------------------------------------------
        full = (self._no_improve_count >= 2 * self.plateau_window)
        self._shake_population(genoms, scores, full=full)
