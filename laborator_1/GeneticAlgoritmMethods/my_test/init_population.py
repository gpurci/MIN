#!/usr/bin/python

import numpy as np
from my_code.init_population import *
from my_code.metrics import *


class TestInitPopulation(InitPopulation):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, population size.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        metrics = self._build_test_metrics()
        super().__init__(config, metrics)
        self.GENOME_LENGTH = 5  # local override

    def _build_test_dataset(self):
        coords = np.array([
            [0, 0],
            [3, 0],
            [3, 4],
            [0, 4],
            [1.5, 2]
        ], dtype=np.float64)

        D = np.array([
            [0, 3, 5, 4, 2],
            [3, 0, 4, 5, 3],
            [5, 4, 0, 3, 3],
            [4, 5, 3, 0, 2],
            [2, 3, 3, 2, 0]
        ], dtype=np.float64)

        profit = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        weight = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        return {
            "coords": coords,
            "distance": D,
            "item_profit": profit,
            "item_weight": weight
        }

    def _build_test_metrics(self):

        dataset = self._build_test_dataset()
        metrics = Metrics("TSP")
        metrics.setDataset(dataset)
        metrics.coord = dataset["coords"]
        metrics.GENOME_LENGTH = 5
        return metrics
