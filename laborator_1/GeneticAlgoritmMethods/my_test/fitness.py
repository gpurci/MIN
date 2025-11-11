#!/usr/bin/python

import numpy as np
from my_code.fitness import *
from my_test.root_GA import *
from my_test.metrics import *


class TestFitness(RootGA):
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self):
        super().__init__()

        # mic dataset 5x5 ca sa vedem numere mici
        D = np.array([
            [0,3,5,4,2],
            [3,0,4,5,3],
            [5,4,0,3,3],
            [4,5,3,0,2],
            [2,3,3,2,0]
        ], dtype=np.float64)

        self.D = D

        # dataset TTP mic (1 item/oras)
        dataset_ttp = {
            "coords":       np.zeros((5,2)),  # nu conteaza aici
            "distance":     D,
            "item_profit":  np.array([10,20,30,40,50], dtype=np.float32),
            "item_weight":  np.ones(5, dtype=np.float32),
        }

        self.metrics_tsp = Metrics("TSP")
        self.metrics_tsp.setDataset(D)
        self.metrics_tsp.GENOME_LENGTH = 5

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
        f = Fitness("TSP_f1score")
        f.setMetrics(self.metrics_tsp)
        out = f(self.make_population(), self.metrics_tsp(self.make_population()))
        print(out)


    def test_TTP_linear(self):
        print("\n=== TEST TTP Linear ===")
        f = Fitness("TTP_linear")
        f.setMetrics(self.metrics_ttp)
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
        m   = self.metrics_ttp(pop)
        out = f(pop, m)
        print(out)



    def test_TTP_exp(self):
        print("\n=== TEST TTP Exp ===")
        f = Fitness("TTP_exp")
        f.setMetrics(self.metrics_ttp)
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
        m   = self.metrics_ttp(pop)
        out = f(pop, m)
        print(out)
