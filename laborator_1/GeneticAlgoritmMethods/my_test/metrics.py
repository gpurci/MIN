#!/usr/bin/python

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from laborator_1.GeneticAlgoritmMethods.my_code.metrics import Metrics

class TestMetrics(Metrics):

    def __init__(self):
        super().__init__("TSP")

        # tiny dataset (5 cities)
        coords = np.array([
            [0,0],
            [3,0],
            [3,4],
            [0,4],
            [1.5,2]
        ], dtype=np.float64)

        D = self._pairwise_distance(coords, is_ceil2d=True)
        profit = np.array([10,20,30,40,50], dtype=np.float32)
        weight = np.array([1,2,3,4,5],     dtype=np.float32)

        dataset = {
            "coords"      : coords,
            "distance"    : D,
            "item_profit" : profit,
            "item_weight" : weight
        }

        self.setDataset(dataset)
        self.dataset = dataset["distance"]
        self.GENOME_LENGTH = coords.shape[0]

        self.population = np.array([
            [0,1,2,3,0],
            [0,4,2,1,0],
            [3,2,4,1,3]
        ], dtype=np.int32)


    # --- tests ------------------------------------------

    def test_dataset(self):
        print("\n--- test_dataset ---")
        print("distance[0,2] =", self.dataset[0,2])

    def test_metricsTSP(self):
        print("\n--- test_metricsTSP ---")
        res = self(self.population)
        print("Distances:",   res["distances"])
        print("Num cities:", res["number_city"])

    def test_speedTTP(self):
        print("\n--- test_speedTTP ---")
        for W in [0, 10, 50, 200]:
            v = self.computeSpeedTTP(W, vmax=1.0, vmin=0.1, Wmax=200)
            print(f"W={W:3d} → v={v:.4f}")

    def test_individDistanceTTP(self):
        print("\n--- test_individDistanceTTP ---")
        for ind in self.population:
            dist = self.getIndividDistanceTTP(ind, self.dataset)
            print(ind.tolist(), "→", dist)
