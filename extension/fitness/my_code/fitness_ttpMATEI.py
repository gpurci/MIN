#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *


class TTPFitness(RootGA):
    """
    Fitness operators ONLY for TTP.
    Supports:
        - TTP
        - TTP_f1score
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"DefaultTTPFitnessOps(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """DefaultTTPFitnessOps:
    TTP(size)
    TTP_f1score(size)

    Extern config example:
        fitness = {"method":"extern", "extern_fn":obj, "R":1.0}
"""

    def __unpackMethod(self, method):
        table = {
            "TTP":          self.fitnessTTP,
            "TTP_f1score":  self.fitnessF1scoreTTP,
        }
        return table.get(method, self.fitnessAbstract)

    def __call__(self, metric_values, **call_configs):
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(metric_values, **cfg)

    # ------------------------------------------------------------
    # ERROR
    # ------------------------------------------------------------
    def fitnessAbstract(self, *a, **kw):
        raise NameError(f"Metoda '{self.__method}' nu exista in DefaultTTPFitnessOps")

    # ------------------------------------------------------------
    # FITNESS TTP METHODS 
    # ------------------------------------------------------------

    def fitnessTTP(self, metrics_values):
        """
        Raw TTP fitness (just maximize profit / minimize time).
        """
        return metrics_values.copy()

    def fitnessF1scoreTTP(self, metrics_values, R=1):
        """
        F1-score-inspired TTP combination:
            F = (profit_norm + R * (1/time_norm)) / (R + 1)
        """
        mv = metrics_values.copy()
        mv = self.normalization(mv)
        return (mv + R * mv) / (R + 1)

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
