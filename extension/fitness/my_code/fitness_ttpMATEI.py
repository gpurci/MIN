#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class TTPFitness(RootGA):
    """
    Old-project-compatible TTP fitness, adapted to also match the
    interface/semantics used in the first project.

    Methods:
        - TTP_standard / TTP  : profits - R * times (masked)
        - TTP_f1score         : F1-like score with capacity penalty and alpha
    """

    def __init__(self, method=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"TTPFitness(method={self.__method}, configs={self.__configs})"

    def setParameters(self, **kw):
        super().setParameters(**kw)
        # CHANGED: still allow explicit override of GENOME_LENGTH
        if "GENOME_LENGTH" in kw:
            self.GENOME_LENGTH = kw["GENOME_LENGTH"]

    def __unpackMethod(self, method):
        table = {
            "TTP_standard": self.fitnessTTPStandard,
            "TTP_f1score":  self.fitnessF1scoreTTP,
        }
        return table.get(method, self.fitnessAbstract)

    def __call__(self, metric_values, **call_configs):
        """
        Same pattern as in your first project: configs are stored in
        self.__configs, but can be overridden per-call via **call_configs.
        Then we forward everything into the selected fitness method.
        """
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(metric_values, **cfg)

    def fitnessAbstract(self, *args, **kw):
        raise NameError(f"Fitness method '{self.__method}' does not exist in TTPFitness")

    # ------------------------------------------------------------
    #  TTP_standard == (profit - R*time) * mask_city
    # ------------------------------------------------------------
    # Accept **kw so extra configs don't crash this method.
    def fitnessTTPStandard(self, metric_values, R=1.0, **kw):
        profits     = np.asarray(metric_values["profits"], dtype=np.float64)
        times       = np.asarray(metric_values["times"], dtype=np.float64)
        number_city = np.asarray(metric_values["number_city"], dtype=np.float64)

        # only full tours get non-zero fitness
        mask_city = (number_city >= self.GENOME_LENGTH).astype(np.float64)

        fit = (profits - R * times) * mask_city

        # shift to positive (GA requires positive fitness)
        min_fit = fit.min()
        if min_fit <= 0:
            fit = fit - min_fit + 1e-6
        return fit

    # ------------------------------------------------------------
    #  F1-like TTP fitness with capacity penalty & alpha exponent
    #
    #  CLOSE TO PROJECT-1 Fitness.fitnessF1scoreTTP:
    #     - uses R, W, alpha, beta (optional)
    #     - penalizes overweight solutions
    #     - ensures positive values
    #     - raises to alpha for selective pressure
    #  Here we also optionally use number_obj as an extra factor if present.
    # ------------------------------------------------------------
    def fitnessF1scoreTTP(
        self,
        metric_values,
        R=1.0,
        beta=1.0,
        W=None,
        alpha=None,
        **kw
    ):
        # accepts R, W, alpha, beta, **kw so passing W=..., alpha=...
        profits     = np.asarray(metric_values["profits"], dtype=np.float64)
        times       = np.asarray(metric_values["times"], dtype=np.float64)
        # CHANGED: we also use weights here for overweight penalty
        weights     = np.asarray(metric_values["weights"], dtype=np.float64)
        number_city = np.asarray(metric_values["number_city"], dtype=np.float64)

        # NEW: number_obj is optional; if metrics don't provide it,
        # we just use a factor of 1.0 for everyone.
        if "number_obj" in metric_values:
            number_obj = np.asarray(metric_values["number_obj"], dtype=np.float64)
        else:
            number_obj = np.ones_like(profits, dtype=np.float64)

        # 1) mask for full tours (same idea as project-1 __cityBinaryTSP)
        mask_city = (number_city >= self.GENOME_LENGTH).astype(np.float64)

        # 2) determine capacity Wmax:
        #    - explicit W argument has priority
        #    - otherwise, try self.__configs["W"]
        if W is None:
            Wmax = self.__configs.get("W", None)
        else:
            Wmax = W

        if Wmax is None:
            raise ValueError(
                "TTPFitness: missing capacity 'W'. "
                "Provide W either in constructor or per-call."
            )

        # 3) overweight penalty like in project-1 Fitness.fitnessF1scoreTTP
        overweight = np.maximum(0.0, weights - Wmax)
        # allow beta from __configs if provided
        beta_cfg = self.__configs.get("beta", None)
        if beta_cfg is not None:
            beta = beta_cfg
        penalty = beta * overweight

        # 4) base F1-like term, similar to:
        #       2 * profit / (profit + R * time + penalty)
        #    multiplied by:
        #       mask_city  (only full tours)
        #       number_obj (optional extra factor from metrics)
        raw = mask_city * number_obj * (
            2.0 * profits / (profits + R * times + penalty + 1e-9)
        )

        # 5) shift to positive (GA needs > 0 fitness)
        min_raw = raw.min()
        if min_raw <= 0:
            raw = raw - min_raw + 1e-6

        # 6) selective pressure exponent alpha
        if alpha is None:
            alpha = self.__configs.get("alpha", 2.0)

        return raw ** alpha

    def normalization(self, x):
        """
        OLD inverted min-max normalization. Kept for backwards
        compatibility; not used by the new F1score formula above,
        but safe to leave in place.
        """
        x = np.asarray(x, dtype=np.float64)
        x_min = x.min()
        x_max = x.max()
        return (x_max - x) / (x_max - x_min + 1e-7)

    def min_norm(self, x):
        """
        OLD min_norm used in historical TTP fitness. We keep it
        here so any legacy code that relied on it still works.
        """
        x = np.asarray(x, dtype=np.float64)
        mask_not_zero = (x != 0)
        valid = x[mask_not_zero]

        if valid.size > 0:
            m = valid.min()
        else:
            m = 0.1
            x[:] = 0.1

        return (2 * m) / (x + m)
