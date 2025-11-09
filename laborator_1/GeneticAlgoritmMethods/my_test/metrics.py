#!/usr/bin/python
import numpy as np
from my_code.metrics import Metrics

class TestMetrics:

    def __init__(self):

        # ===== tiny TSP data =====
        D_tsp = np.array([
            [0,3,5,4,2],
            [3,0,4,5,3],
            [5,4,0,3,3],
            [4,5,3,0,2],
            [2,3,3,2,0]
        ], dtype=np.float64)

        self.metrics_tsp = Metrics("TSP")
        self.metrics_tsp.setDataset(D_tsp)
        self.metrics_tsp.GENOME_LENGTH = 5

        self.population_tsp = np.array([
            [0,1,2,3,0],
            [0,4,2,1,0],
        ], dtype=np.int32)


        # ===== tiny TTP data =====
        profit = np.array([10,20,30,40,50], dtype=np.float32)
        weight = np.array([1,2,3,4,5],     dtype=np.float32)

        dataset_ttp = {
            "coords":       np.zeros((5,2)),
            "distance":     D_tsp,
            "item_profit":  profit,
            "item_weight":  weight,
        }

        self.metrics_ttp = Metrics("TTP")
        self.metrics_ttp.setDataset(dataset_ttp)
        self.metrics_ttp.GENOME_LENGTH = 5
        self.metrics_ttp.v_max = 1.0
        self.metrics_ttp.v_min = 0.1
        self.metrics_ttp.W = 200.0

        self.population_ttp = np.array([
            [0,1,2,3,4,0],
            [3,2,4,1,0,3],
        ], dtype=np.int32)



    # ================ tests common ================
    def test_help(self):
        print("\n=== test_help ===")
        print("TSP ->", self.metrics_tsp.help())
        print("TTP ->", self.metrics_ttp.help())


    # ================== TSP =======================
    def test_metricsTSP(self):
        print("\n=== test_metricsTSP ===")
        res = self.metrics_tsp(self.population_tsp)
        print(res)

    def test_scoreTSP(self):
        print("\n=== test_scoreTSP ===")
        fit = np.array([0.1,0.9])
        r = self.metrics_tsp.getScore(self.population_tsp, fit)
        print(r)


    # ================== TTP =======================
    def test_metricsTTP(self):
        print("\n=== test_metricsTTP ===")
        res = self.metrics_ttp(self.population_ttp)
        print(res)

    def test_speedTTP(self):
        print("\n=== test_speedTTP ===")
        for W in [0,3,10]:
            v = self.metrics_ttp.computeSpeedTTP(W,1.0,0.1,20)
            print("W=",W,"→ v=",v)

    def test_scoreTTP(self):
        print("\n=== test_scoreTTP ===")
        fit = np.array([0.2,0.8])
        r = self.metrics_ttp.getScore(self.population_ttp, fit)
        print(r)
