#!/usr/bin/python
import numpy as np
from my_code.metrics import *

class TestMetrics(Metrics):

    def __init__(self, **conf):
        # config must be "tsp" to activate metricsTSP
        # este test general trebuie de specificat metodele separat 
        # sau de creat functii separate care testeaza fiecare functional
        super().__init__("tsp")

    def test(self):
        np.random.seed(0)

        # tiny dataset (5 cities)
        coords = np.array([
            [0,0],
            [3,0],
            [3,4],
            [0,4],
            [1.5,2]
        ], dtype=np.float64)

        # distance matrix using ceil2d
        D = self._pairwise_distance(coords, ceil2d=True)

        # dummy profits/weights for 5 nodes
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

        # ---------- population: 3 small routes ----------
        population = np.array([
            [0,1,2,3,0],
            [0,4,2,1,0],
            [3,2,4,1,3]
        ], dtype=np.int32)

        print("\n=== TEST METRICS ===")
        print("Distance matrix:\n", D)
        print("Population:\n", population)

        # compute metrics
        res = self(population)

        print("\nDistances:   ", res["distances"])
        print("Num. cities: ", res["number_city"])

        # recompute pairwise distance on coords
        print("\nPairwise distance test:", self._pairwise_distance(coords, True)[0,2])

        # computeSpeedTTP
        print("\n=== TTP speed test ===")
        for W in [0, 10, 50, 200]:
            v = self.computeSpeedTTP(W, vmax=1.0, vmin=0.1, Wmax=200)
            print(f"Wcur = {W:3d}  → v = {v:.4f}")

        # getIndividDistanceTTP
        print("\n=== TTP individ TTP-distance test ===")
        for ind in population:
            dist = self.getIndividDistanceTTP(ind, D)
            print(f"individ {ind.tolist()}  → TTP distance = {dist}")
