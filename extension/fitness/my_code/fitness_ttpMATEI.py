#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class TTPFitness(RootGA):
    """
    Fitness operators ONLY for TTP.
    Supports:
        - TTP_standard  (profit - R * time)
        - TTP           (alias)
        - TTP_f1score   (normalized composite)
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"TTPFitness(method={self.__method}, configs={self.__configs})"

    # --------------------------------------------------------------------
    # HELP
    # --------------------------------------------------------------------
    def help(self):
        return """TTPFitness:
    Methods:
        TTP_standard(R=1)
        TTP           (alias of TTP_standard)
        TTP_f1score(R=1)

    Extern example:
        fitness = {"method":"extern", "extern_fn":obj, "R":1.0}
"""

    # --------------------------------------------------------------------
    # DISPATCH
    # --------------------------------------------------------------------
    def __unpackMethod(self, method):
        table = {
            "TTP":          self.fitnessTTP,       # alias
            "TTP_standard": self.fitnessTTP,
            "TTP_f1score":  self.fitnessF1scoreTTP,
        }
        return table.get(method, self.fitnessAbstract)

    # --------------------------------------------------------------------
    def __call__(self, metric_values, **call_configs):
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(metric_values, **cfg)

    def fitnessAbstract(self, *args, **kw):
        raise NameError(f"Fitness method '{self.__method}' does not exist in TTPFitness")

    # --------------------------------------------------------------------
    # MAIN RAW FITNESS: profit - R * time
    # --------------------------------------------------------------------
    def fitnessTTP(self, metrics_values, R=1.0, shift_positive=True):
        """
        Classical TTP objective:
            fitness = profit - R * time

        IMPORTANT:
        Early generations often produce very negative values.
        To avoid GA freezing, we shift them to be >= epsilon,
        without changing the relative ordering.
        """
        profit = np.asarray(metrics_values["profit"], dtype=np.float64)
        time   = np.asarray(metrics_values["time"], dtype=np.float64)

        raw = profit - R * time

        if shift_positive:
            raw = raw - raw.min() + 1e-12  # preserves ordering

        return raw

    # --------------------------------------------------------------------
    # NORMALIZED F1-LIKE FITNESS
    # --------------------------------------------------------------------
    def fitnessF1scoreTTP(self, metrics_values, R=1.0):
        """
        Combines normalized profit (high=good) and inverse time (low=good).
        """
        profit = np.asarray(metrics_values["profit"], dtype=np.float64)
        time   = np.asarray(metrics_values["time"], dtype=np.float64)

        # normalize profit → [0, 1]
        p = self.normalization(profit)

        # normalize time (low→1, high→0)
        t = self.min_norm(time)

        return (p + R * t) / (1.0 + R)

    # --------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------
    def normalization(self, x):
        x = np.asarray(x, dtype=np.float64)
        xmin = x.min()
        xmax = x.max()
        if xmax == xmin:
            return np.ones_like(x)
        return (x - xmin) / (xmax - xmin)

    def min_norm(self, x):
        """
        Convert time such that:
            small time -> score ~1
            large time -> score ~0
        """
        x = np.asarray(x, dtype=np.float64)

        mask = (x != 0)
        valid = x[mask]

        if valid.size == 0:
            m = 0.1
            x = np.full_like(x, 0.1)
        else:
            m = valid.min()

        return (2 * m) / (x + m)
