import numpy as np
from GeneticAlgorithmManager.my_code.algoritm_genetic import GeneticAlgorithm
from GeneticAlgorithmManager.my_code.genoms import Genoms

class GeneticAlgorithmWithEliteSearch(GeneticAlgorithm):

    def __init__(self, *args, elite_search=None, elite_chrom="tsp",
                 verbose_elite=True, **configs):

        self.elite_search  = elite_search
        self.elite_chrom   = elite_chrom
        self.verbose_elite = verbose_elite
        self._gene_range   = configs.get("genoms", {}).copy()

        # generation counter (safe, subclass-owned)
        self._internal_generation = -1

        super().__init__(*args, **configs)

    def setParameters(self, **kw):
        super().setParameters(**kw)

        if self.elite_search:
            self.elite_search.setParameters(**kw)

        # how often to run elite local search
        self.elite_freq = kw.get("ELITE_FREQ", 20)

    # ---------------------------
    # Improve elites using LS
    # ---------------------------
    def _improve_elites(self, elites):

        if self.elite_search is None:
            return elites

        elites2 = elites.copy()
        n_elites = elites.shape[0]

        # ---------------------------------------------------------
        # FULL GENOME LOCAL SEARCH (TTP-aware VND)
        # ---------------------------------------------------------
        if self.elite_chrom == "":  
            for i in range(n_elites):
                elites2[i] = self.elite_search(None, None, elites2[i])

            if self.verbose_elite:
                print(f"[EliteSearch] ✓ Improved {n_elites}/{n_elites} elites")

            return elites2

        # ---------------------------------------------------------
        # CLASSIC SINGLE-CHROMOSOME LOCAL SEARCH (TSP-only)
        # ---------------------------------------------------------
        for i in range(n_elites):
            route = elites2[i][self.elite_chrom].copy()
            elites2[i][self.elite_chrom] = self.elite_search(None, None, route)

        if self.verbose_elite:
            print(f"[EliteSearch] ✓ Improved {n_elites}/{n_elites} elites")

        return elites2


    # ---------------------------------
    # Compute fitness of modified elites
    # ---------------------------------
    def _fitness_of_elites(self, elites):
        tmp = Genoms(size=self.GENOME_LENGTH, **self._gene_range)

        for e in elites:
            tmp.append(e)

        tmp.save()

        old = self.POPULATION_SIZE
        self.metrics.setParameters(POPULATION_SIZE=elites.shape[0])
        f = self.fitness(self.metrics(tmp))
        self.metrics.setParameters(POPULATION_SIZE=old)

        return f

    # ---------------------------------------------------------------------
    # Override ONLY elite insertion — run local search every elite_freq gens
    # ---------------------------------------------------------------------
    def setElitesByFitness(self, fitness_values, elites, fitness_elites=None):

        # Called once per generation → safe counter
        self._internal_generation += 1

        run_ls = (self._internal_generation % self.elite_freq == 0)

        elites_out = elites.copy()

        # run LS every X generations
        if run_ls and self.elite_search is not None:
            elites_out = self._improve_elites(elites_out)
            fitness_elites = self._fitness_of_elites(elites_out)

        # call original GA method
        return super().setElitesByFitness(
            fitness_values,
            elites_out,
            fitness_elites
        )
