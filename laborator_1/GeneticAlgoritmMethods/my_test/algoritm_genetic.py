#!/usr/bin/python

import numpy as np
import sys
from pathlib import Path

# get path: .../Homeworks/MIN
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from laborator_1.GeneticAlgoritmMethods.my_code.algoritm_genetic import GeneticAlgorithm

class TestGeneticAlgorithm(object):

    def __init__(self, name="testGA"):
        self.name = name

    def test_run(self):
        configs = {
            "metric": "TSP",
            "init_population": "TSP_aleator",
            "fitness": "TSP_f1score",
            "select_parent": {
                "select_parent1": "choise",
                "select_parent2": "choise"
            },
            "crossover": "diff",
            "mutate": "swap",
            "repair": None,
            "callback": "test_ga_log.csv"
        }
        ga = GeneticAlgorithm(self.name, **configs)
        ga.setParameters(POPULATION_SIZE=5, GENOME_LENGTH=5, GENERATIONS=2)

        D = np.array([
            [0,3,4,3,2],
            [3,0,4,5,2],
            [4,4,0,3,3],
            [3,5,3,0,3],
            [2,2,3,3,0],
        ], dtype=np.float64)

        dataset = {
            "coords": np.zeros((5,2)),
            "distance": D,
            "item_profit": np.ones(5, dtype=np.float32),
            "item_weight": np.ones(5, dtype=np.float32)
        }

        ga.setDataset(dataset)
        best_individ, final_population = ga(population=None)

        print("=== TestGeneticAlgorithm done ===")
        print("best_individ:", best_individ)
        print("final_population:\n", final_population)
