#!/usr/bin/python

import numpy as np
from laborator_1.GeneticAlgoritmMethods.my_code.root_GA import RootGA
from laborator_1.GeneticAlgoritmMethods.my_code.metrics import Metrics
from laborator_1.GeneticAlgoritmMethods.my_code.fitness import Fitness

class TestFitness(RootGA):

    def __init__(self):
        super().__init__()

        D = np.array([
            [0,3,5,4,2],
            [3,0,4,5,3],
            [5,4,0,3,3],
            [4,5,3,0,2],
            [2,3,3,2,0]
        ], dtype=np.float64)

        dataset = {
            "coords":       np.zeros((5,2)),
            "distance":     D,
            "item_profit":  np.arange(10,60,10, dtype=np.float32),
            "item_weight":  np.ones(5, dtype=np.float32),
        }

        self.metrics = Metrics("TSP")
        self.metrics.setDataset(dataset)
        self.metrics.GENOME_LENGTH = 5

        self.D = D

    def make_population(self):
        return np.array([
            [0,1,2,3,4,0],
            [0,4,3,2,1,0],
        ], dtype=np.int32)

    def test_TSP_f1(self):
        f = Fitness("TSP_f1score", self.metrics)
        print("\nTEST TSP F1score")
        print(f(self.make_population()))

    def test_TTP_linear(self):
        f = Fitness("TTP_linear", self.metrics)
        f.setTTPParams(distance=self.D, items=[(i,1,10) for i in range(5)], 
                       v_min=0.1, v_max=1.0, W=10.0, R=1.0, lam=0.01)
        f.distance = self.D
        f.items = [(i,1,10) for i in range(5)]
        print("\nTEST TTP Linear")
        print(f(self.make_population()))

    def test_TTP_exp(self):
        f = Fitness("TTP_exp", self.metrics)
        f.setTTPParams(distance=self.D, items=[(i,1,10) for i in range(5)], 
                       v_min=0.1, v_max=1.0, W=10.0, R=1.0, lam=0.01)
        f.distance = self.D
        f.items = [(i,1,10) for i in range(5)]
        print("\nTEST TTP Exponential")
        print(f(self.make_population()))
