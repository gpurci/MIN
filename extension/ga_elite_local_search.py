import numpy as np
from GeneticAlgorithmManager.my_code.algoritm_genetic import GeneticAlgorithm
from GeneticAlgorithmManager.my_code.genoms import Genoms

class GeneticAlgorithmWithEliteSearch(GeneticAlgorithm):

    def __init__(self, *args, elite_search=None, elite_chrom="tsp",
                 verbose_elite=False, **configs):
        """
        verbose_elite:
            False → silent (default)
            True  → print messages when tabu runs on elites
        """
        self.elite_search = elite_search
        self.elite_chrom  = elite_chrom
        self.verbose_elite = verbose_elite

        self._gene_range = configs.get("genoms", {}).copy()
        super().__init__(*args, **configs)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if self.elite_search:
            self.elite_search.setParameters(**kw)

    #   Improve Elites (Apply Local Search)
    def _improve_elites(self, elites):
        if self.elite_search is None:
            return elites

        elites2 = elites.copy()
        n = elites2.shape[0]

        for i in range(n):
            route = elites2[i][self.elite_chrom].copy()
            elites2[i][self.elite_chrom] = self.elite_search(route)

        if self.verbose_elite:
            print(f"[EliteSearch] ✓ Improved {n}/{n} elites")

        return elites2


    #   Compute Fitness of Modified Elites
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

    #   Override: This Method Inserts Elites Back Into the GA
    def setElitesByFitness(self, fitness_values, elites, fitness_elites=None):
        elites_imp = self._improve_elites(elites)

        if self.elite_search is not None:
            fitness_elites = self._fitness_of_elites(elites_imp)

        return super().setElitesByFitness(
            fitness_values, elites_imp, fitness_elites
        )
