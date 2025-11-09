#!/usr/bin/python

import numpy as np
from my_code.algoritm_genetic import GeneticAlgorithm
from my_code.metrics import Metrics


class TestGeneticAlgorithm:

    def __init__(self):
        cfg = {
            "metric": "TSP",
            "init_population": "TSP_aleator",
            "fitness": "TSP_f1score",
            "select_parent": {
                "select_parent1": "choice",
                "select_parent2": "choice"
            },
            "crossover": "diff",
            "mutate": "swap",
            "repair": None,
            "callback": None
        }

        self.ga = GeneticAlgorithm("unitGA", **cfg)
        self.ga.setParameters(
            POPULATION_SIZE=5,
            GENOME_LENGTH=5,
            GENERATIONS=1,
            ELITE_SIZE=1
        )

        D = np.array([
            [0,3,4,3,2],
            [3,0,4,5,2],
            [4,4,0,3,3],
            [3,5,3,0,3],
            [2,2,3,3,0],
        ], dtype=np.float64)

        self.ga.setDataset(D)

        self.pop = np.array([
            [0, 1, 2, 3, 4],
            [1, 4, 3, 2, 0],
            [2, 3, 0, 4, 1],
            [3, 1, 4, 0, 2],
            [4, 2, 1, 3, 0],
        ], dtype=np.int32)

        self.metric_values = self.ga.metrics(self.pop)
        self.fit = self.ga.fitness(self.pop, self.metric_values)

    def test_run(self):
        print("\n=== test_run (__call__) ===")
        best, final_pop = self.ga(None)
        print("best:", best)


    def test_help(self):
        print("\n=== test_help ===")
        self.ga.help()

    def test_setDataset(self):
        print("\n=== test_setDataset ===")
        self.ga.metrics = Metrics("TTP")  # ADD THIS LINE
        self.ga.setDataset({"coords": np.zeros((5, 2)), "distance": np.eye(5),
                            "item_profit": np.ones(5), "item_weight": np.ones(5)})
        print("OK setDataset")


    def test_setParameters(self):
        print("\n=== test_setParameters ===")
        self.ga.setParameters(POPULATION_SIZE=7)
        print("POPULATION_SIZE:", self.ga.POPULATION_SIZE)


    def test_evolutionMonitor(self):
        print("\n=== test_evolutionMonitor ===")
        fake = {"score":1.0,"best_fitness":1.0}
        self.ga.evolutionMonitor(fake)
        print("OK evolutionMonitor")

    def test_setElites(self):
        print("\n=== test_setElites ===")
        elites = self.pop[:1]
        pop_closed = self.pop.copy()
        elites_closed = elites.copy()
        out = self.ga.setElites(pop_closed.copy(), elites_closed)
        print(out)

    def test_setElitesByFitness(self):
        print("\n=== test_setElitesByFitness ===")
        elites = self.pop[:1]
        out = self.ga.setElitesByFitness(self.pop.copy(), self.fit, elites)
        print(out)

    def test_showMetrics(self):
        print("\n=== test_showMetrics ===")
        self.ga.showMetrics(0, {"score":1.23,"best_fitness":9.99})


    def test_stres(self):
        print("\n=== test_stres ===")
        self.ga.stres({"score":1.0})
        print("OK stres")


    def test_getArgsWeaks(self):
        print("\n=== test_getArgsWeaks ===")
        print(self.ga.getArgsWeaks(self.fit, 2))


    def test_getArgsElite(self):
        print("\n=== test_getArgsElite ===")
        print(self.ga.getArgsElite(self.fit))
