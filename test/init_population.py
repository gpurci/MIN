#!/usr/bin/python
import numpy as np

from my_code.metrics         import Metrics
from my_code.init_population import InitPopulation

class TestInitPopulation:

    def __init__(self):
        # mic dataset pt TTP
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
            "item_profit":  np.array([10,20,30,40,50]),
            "item_weight":  np.ones(5),
        }

        self.metrics = Metrics("TTP")
        self.metrics.setDataset(dataset)
        self.metrics.GENOME_LENGTH = 5

    # ---------------------------------------------------------
    def test_help(self):
        print("\n=== test_help ===")
        ip = InitPopulation("vecin", self.metrics)
        print(ip.help())

    # ---------------------------------------------------------
    def test_abstract(self):
        print("\n=== test_abstract ===")
        ip = InitPopulation("unknown", self.metrics)
        try:
            ip(5)
        except NameError as e:
            print("OK abstract throws:", e)

    # ---------------------------------------------------------
    def test_TSP_random(self):
        print("\n=== test_TSP_random ===")
        ip = InitPopulation("TSP_aleator", self.metrics)
        ip.GENOME_LENGTH = 5
        pop = ip(10)
        print("shape =", pop.shape)
        print(pop[:3])

    # ---------------------------------------------------------
    def test_init_TTP(self):
        print("\n=== test_init_TTP ===")
        ip = InitPopulation("vecin", self.metrics)
        ip.GENOME_LENGTH = 5
        pop = ip.initPopulationTTP(size=10, lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=20, seed=0)
        print("shape =", pop.shape)
        print(pop[:3])

        print("\n--- internal _constructGreedyRoute ---")
        r1 = ip._constructGreedyRoute(start=0, lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=20)
        print(r1)

        print("\n--- internal _twoOpt ---")
        bad = np.array([0, 4, 3, 2, 1, 0], dtype=np.int32)
        r2 = ip._twoOpt(bad)
        print("before:", bad)
        print("after :", r2)
