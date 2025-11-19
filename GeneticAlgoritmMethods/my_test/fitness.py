#!/usr/bin/python

import  numpy as np
from my_code.fitness import *
from my_test.root_GA import *
from my_test.metrics import *

class TestFitness(RootGA):
    def __init__(self):
        super().__init__()

        # mic dataset 5x5 ca să vedem numere mici
        D = np.array([
            [0,3,5,4,2],
            [3,0,4,5,3],
            [5,4,0,3,3],
            [4,5,3,0,2],
            [2,3,3,2,0]
        ], dtype=np.float64)

        self.D = D

        # dataset TTP (1 item/oras)
        dataset_ttp = {
            "coords":       np.zeros((5,2)),
            "distance":     D,
            "item_profit":  np.array([10,20,30,40,50], dtype=np.float32),
            "item_weight":  np.ones(5, dtype=np.float32),
        }

        # metrics TSP
        self.metrics_tsp = Metrics("TSP")
        self.metrics_tsp.setDataset(D)
        self.metrics_tsp.GENOME_LENGTH = 5

        # metrics TTP
        self.metrics_ttp = Metrics("TTP")
        self.metrics_ttp.setDataset(dataset_ttp)
        self.metrics_ttp.GENOME_LENGTH = 5
        self.metrics_ttp.v_min = 0.1
        self.metrics_ttp.v_max = 1.0
        self.metrics_ttp.W     = 10.0


    def make_population(self):
        return np.array([
            [0,1,2,3,4,0],
            [0,4,3,2,1,0],
        ], dtype=np.int32)


    def test_TSP_f1(self):
        print("\n=== TEST TSP F1score ===")
        f   = Fitness("TSP_f1score")

        pop = self.make_population()
        m   = self.metrics_tsp(pop)      # ← calculăm metrici înainte
        out = f(pop, m)                  # ← trimitem metrici în call

        print(out)


    def test_TTP_linear(self):
        print("\n=== TEST TTP Linear ===")
        f = Fitness("TTP_linear")

        f.setTTPParams(
            distance = self.metrics_ttp.distance,
            items    = self.metrics_ttp.items,
            v_min    = 0.1,
            v_max    = 1.0,
            W        = 10.0,
            R        = 1.0,
            lam      = 0.00,
            alpha    = 0.1
        )

        pop = self.make_population()
        m   = self.metrics_ttp(pop)      # ← calculăm metrici înainte
        out = f(pop, m)

        print(out)


    def test_TTP_exp(self):
        print("\n=== TEST TTP Exp ===")
        f = Fitness("TTP_exp")

        f.setTTPParams(
            distance = self.metrics_ttp.distance,
            items    = self.metrics_ttp.items,
            v_min    = 0.1,
            v_max    = 1.0,
            W        = 10.0,
            R        = 1.0,
            lam      = 0.01   # exponential decay
        )

        pop = self.make_population()
        m   = self.metrics_ttp(pop)      # ← calculăm metrici înainte
        out = f(pop, m)

        print(out)
