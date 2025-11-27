#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.algoritm_genetic import GeneticAlgorithm
from GeneticAlgorithmManager.my_code.genoms import Genoms


class GeneticAlgorithmWithEliteSearch(GeneticAlgorithm):
    """
    Genetic Algorithm with:
      - Elite local search (e.g., TTPVNDLocalSearch)
      - Optional adaptive selection rate over generations
      - Optional uniqueness filtering for elites
    """

    def __init__(
        self,
        *args,
        elite_search=None,
        elite_chrom="tsp",
        verbose_elite=True,
        **configs,
    ):
        # --- elite LS configuration ---
        self.elite_search  = elite_search
        self.elite_chrom   = elite_chrom   # "tsp", "kp", or "" for full TTP genome
        self.verbose_elite = verbose_elite

        # gene range for temporary Genoms in _fitness_of_elites
        # (this is the dict you pass as `genoms={...}` in builder_ttp.py)
        self._gene_range   = (configs.get("genoms", {}) or {}).copy()

        # safe generation counter (we control it)
        self._internal_generation = -1

        # ---- selection schedule parameters (from manager dict) ----
        manager_cfg = configs.get("manager", {}) or {}

        # OPTIONAL: if not provided, selection stays static
        self._select_rate_min       = manager_cfg.get("select_rate_min", None)
        self._select_rate_max       = manager_cfg.get("select_rate_max", None)
        self._select_rate_schedule  = manager_cfg.get("select_rate_schedule", "linear")

        # will be set in setParameters() once SELECT_RATE is known
        self._select_rate0 = None

        super().__init__(*args, **configs)

    # ---------------------------------------------------------
    # Standard parameter setup + wiring elite_search + elite_freq
    # ---------------------------------------------------------
    def setParameters(self, **kw):
        super().setParameters(**kw)

        # propagate params to elite local search operator
        if self.elite_search:
            self.elite_search.setParameters(**kw)

        # how often to run elite local search
        self.elite_freq = kw.get("ELITE_FREQ", getattr(self, "elite_freq", 20))

        # remember original SELECT_RATE once (for reference / debugging)
        if self._select_rate0 is None and hasattr(self, "SELECT_RATE"):
            self._select_rate0 = self.SELECT_RATE

    # ---------------------------------------------------------
    # Internal: update SELECT_RATE according to schedule
    # ---------------------------------------------------------
    def _update_select_rate(self):
        """
        Adaptive SELECT_RATE schedule:
            - if select_rate_min/max are provided via manager,
              linearly interpolate from max at gen=0 to min at gen=GENERATIONS-1.
        """
        if self._select_rate_min is None or self._select_rate_max is None:
            # no schedule requested
            return

        if not hasattr(self, "GENERATIONS") or self.GENERATIONS is None:
            return

        G = max(1, int(self.GENERATIONS))
        g = max(0, min(self._internal_generation, G - 1))

        # linear schedule: start high (max) → end low (min)
        if self._select_rate_schedule == "linear":
            start = float(self._select_rate_max)
            end   = float(self._select_rate_min)
            if G <= 1:
                new_rate = end
            else:
                t = g / float(G - 1)
                new_rate = start + (end - start) * t
        else:
            # unknown schedule type → do nothing
            return

        # clamp to sane range
        new_rate = max(0.05, min(0.95, new_rate))
        self.SELECT_RATE = new_rate

    # ---------------------------
    # Improve elites using LS
    # ---------------------------
    def _improve_elites(self, elites):
        """
        Apply local search to elites.

        If elite_chrom == "":
            treat each elite as full TTP genome and call elite_search on it.

        Else:
            treat elite_chrom (e.g. "tsp") as a route chromosome
            and apply TSP-only local search.
        """
        if self.elite_search is None:
            return elites

        elites2 = elites.copy()
        n_elites = elites.shape[0]

        # FULL GENOME LOCAL SEARCH (TTP-aware VND)
        if self.elite_chrom == "":
            for i in range(n_elites):
                elites2[i] = self.elite_search(None, None, elites2[i])

            if self.verbose_elite:
                print(f"[EliteSearch] ✓ Improved {n_elites}/{n_elites} elites (full genome)")

            return elites2

        # SINGLE-CHROMOSOME LOCAL SEARCH (e.g. TSP route only)
        for i in range(n_elites):
            route = elites2[i][self.elite_chrom].copy()
            elites2[i][self.elite_chrom] = self.elite_search(None, None, route)

        if self.verbose_elite:
            print(f"[EliteSearch] ✓ Improved {n_elites}/{n_elites} elites (chrom='{self.elite_chrom}')")

        return elites2

    # ---------------------------------
    # Compute fitness of modified elites
    # ---------------------------------
    def _fitness_of_elites(self, elites):
        """
        Re-evaluate fitness of elites after local search by
        building a temporary Genoms object and calling fitness().
        """
        tmp = Genoms(
            genome_lenght=self.GENOME_LENGTH,
            **self._gene_range
        )

        for e in elites:
            tmp.append(e)

        tmp.save()

        old = self.POPULATION_SIZE
        self.metrics.setParameters(POPULATION_SIZE=elites.shape[0])
        f = self.fitness(self.metrics(tmp))
        self.metrics.setParameters(POPULATION_SIZE=old)

        return f

    # ------------------------------------------------------------------
    # Ensure elites are unique (no duplicate chromosomes)
    # ------------------------------------------------------------------
    def _unique_elites(self, elites, fitness_elites=None):
        """
        Remove duplicate elite chromosomes based on their raw bytes.

        elites:         np.ndarray of chromosomes (Genoms dtype)
        fitness_elites: None or 1D array-like aligned with elites
        """
        if elites is None:
            return elites, fitness_elites

        elites = np.asarray(elites)
        if elites.size == 0:
            return elites, fitness_elites

        seen = set()
        kept_indices = []

        for i, chrom in enumerate(elites):
            key = chrom.tobytes()  # works for structured dtypes too
            if key in seen:
                continue
            seen.add(key)
            kept_indices.append(i)

        if len(kept_indices) == len(elites):
            return elites, fitness_elites

        elites_unique = elites[kept_indices]

        if fitness_elites is None:
            return elites_unique, None

        fitness_elites = np.asarray(fitness_elites)
        fitness_unique = fitness_elites[kept_indices]
        return elites_unique, fitness_unique

    # ---------------------------------------------------------------------
    # Override ONLY elite insertion — run LS every elite_freq generations
    # and update SELECT_RATE schedule once per generation.
    # ---------------------------------------------------------------------
    def setElitesByFitness(self, fitness_values, elites, fitness_elites=None):
        """
        Called once per generation by the base GA.

        We hook here to:
          - bump internal generation counter
          - update SELECT_RATE adaptively
          - occasionally run elite local search
          - ensure elites are unique
        """
        # generation counter we control
        self._internal_generation += 1

        # Sync mutation rate from stress → GA (but only if increased)
        if hasattr(self.stres, "MUTATION_RATE") and self.stres.MUTATION_RATE > self.MUTATION_RATE:
            new_rate = float(self.stres.MUTATION_RATE)
            print(f"[GA] Sync MUTATION_RATE (stress → GA): {self.MUTATION_RATE:.4f} → {new_rate:.4f}")
            self.MUTATION_RATE = new_rate

            try:
                self.mutate.setParameters(MUTATION_RATE=new_rate)
            except Exception:
                pass

        # update SELECT_RATE for *next* generation
        self._update_select_rate()

        # whether to run elite LS on this generation
        run_ls = (self._internal_generation % self.elite_freq == 0)

        elites_out = elites.copy()

        if run_ls and self.elite_search is not None:
            elites_out = self._improve_elites(elites_out)
            fitness_elites = self._fitness_of_elites(elites_out)

        elites_out, fitness_elites = self._unique_elites(elites_out, fitness_elites)

        return super().setElitesByFitness(
            fitness_values,
            elites_out,
            fitness_elites
        )
