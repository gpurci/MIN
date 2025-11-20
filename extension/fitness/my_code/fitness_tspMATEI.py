#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *


class TSPFitness(RootGA):
    """
    Fitness operators ONLY for TSP.
    Supports:
        - TSP_norm
        - TSP_f1score
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"DefaultTSPFitnessOps(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """DefaultTSPFitnessOps:
    TSP_norm(size)
    TSP_f1score(size)

    Extern config example:
        fitness = {"method":"extern", "extern_fn":obj}
"""

    def __unpackMethod(self, method):
        table = {
            "TSP_norm":     self.fitnessNormTSP,
            "TSP_f1score":  self.fitnessF1scoreTSP,
        }
        return table.get(method, self.fitnessAbstract)

    def __call__(self, metrics_values, **call_configs):
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(metrics_values, **cfg)

    # ------------------------------------------------------------
    # ERROR
    # ------------------------------------------------------------
    def fitnessAbstract(self, *a, **kw):
        raise NameError(f"Metoda '{self.__method}' nu exista in DefaultTSPFitnessOps")

    # ------------------------------------------------------------
    # FITNESS TSP METHODS
    # ------------------------------------------------------------
    def fitnessNormTSP(self, metrics_values):
        """
        Normalize TSP distances using min-max scaling.
        """
        M = metrics_values.copy()
        M = self.normalization(M)
        return M

    def fitnessF1scoreTSP(self, metrics_values, R=1):
        """
        F1-score-like fitness for TSP (higher=fitter).
        """
        M = metrics_values.copy()
        scores = self.summary(M)
        scores = self.normalization(scores)
        return (scores + R * scores) / (R + 1)  # combine

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def summary(self, x):
        return np.sum(x, axis=-1)

    def normalization(self, x):
        x = np.array(x, dtype=np.float64)
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax == xmin:
            return np.ones_like(x)
        return (x - xmin) / (xmax - xmin)

    def min_norm(self, x):
        mask = (x != 0)
        valid = x[mask]
        if valid.shape[0] > 0:
            m = valid.min()
        else:
            m = 0.1
            x[:] = 0.1
        return (2 * m) / (x + m)
