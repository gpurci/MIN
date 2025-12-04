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
        # keep all constructor configs (R, W, alpha, beta, etc.)
        self.__configs = dict(configs)
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"TTPFitness(method={self.__method}, configs={self.__configs})"

    def setParameters(self, **kw):
        """
        Called from GA.setParameters(...).
        Keeps compatibility with your previous project:
          - can override GENOME_LENGTH
          - can inject dataset, R, W, alpha, beta, ...
        """
        super().setParameters(**kw)

        # still allow explicit override of GENOME_LENGTH
        if "GENOME_LENGTH" in kw:
            self.GENOME_LENGTH = kw["GENOME_LENGTH"]

        # remember dataset & scalar configs if passed here
        if "dataset" in kw:
            self.dataset = kw["dataset"]

        for key in ("R", "W", "alpha", "beta"):
            if key in kw:
                self.__configs[key] = kw[key]

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
        raise NameError(
            f"Fitness method '{self.__method}' does not exist in TTPFitness"
        )

    # ------------------------------------------------------------
    #  Helpers to resolve R and W (capacity) consistently
    # ------------------------------------------------------------
    def _resolve_R(self, R):
        """
        Resolve renting rate R with the following priority:
          1) explicit argument R
          2) constructor / setParameters configs
          3) dataset["R"], if available
          4) fallback to 5.61 (last resort)
        """
        if R is not None:
            return float(R)

        # from stored configs
        if "R" in self.__configs:
            return float(self.__configs["R"])

        # from dataset, if attached via setParameters(dataset=...)
        ds = getattr(self, "dataset", None)
        if isinstance(ds, dict) and "R" in ds:
            return float(ds["R"])

        # last-resort fallback (avoid crashes, but you probably want to override)
        return 5.61

    def _resolve_W(self, W):
        """
        Resolve capacity W (Wmax) similarly:
          1) explicit arg W
          2) configs["W"]
          3) dataset["W"] or dataset["capacity"]
          4) otherwise raise (no silent wrong default)
        """
        if W is not None:
            return float(W)

        if "W" in self.__configs:
            return float(self.__configs["W"])

        ds = getattr(self, "dataset", None)
        if isinstance(ds, dict):
            if "W" in ds:
                return float(ds["W"])
            if "capacity" in ds:
                return float(ds["capacity"])

        raise ValueError(
            "TTPFitness: missing capacity 'W'. "
            "Provide W either in constructor, setParameters, or dataset."
        )

    # ------------------------------------------------------------
    #  TTP_standard == (profit - R*time) * mask_city
    # ------------------------------------------------------------
    # Accept **kw so extra configs don't crash this method.
    def fitnessTTPStandard(self, metric_values, R=None, W=None, **kw):
        """
        GECCO-compatible fitness:
            - On feasible solutions (weight <= W):  fitness = shifted(profit - R*time)
            - On infeasible solutions (weight > W): fitness = tiny (death penalty)
        The logged 'score' remains profit - R*time, so GECCO comparison is untouched.
        """

        profits     = np.asarray(metric_values["profits"], dtype=np.float64)
        times       = np.asarray(metric_values["times"], dtype=np.float64)
        weights     = np.asarray(metric_values["weights"], dtype=np.float64)
        number_city = np.asarray(metric_values["number_city"], dtype=np.float64)

        # Only full tours are valid
        mask_city = (number_city >= self.GENOME_LENGTH).astype(np.float64)

        # Resolve parameters
        R_use = self._resolve_R(R)
        W_use = self._resolve_W(W)

        # -----------------------------
        # Raw TTP objective (GECCO one)
        # -----------------------------
        raw = profits - R_use * times   # (profit - R * travel_time)

        # -------------------------------------
        # DEATH PENALTY FOR INFEASIBLE SOLUTIONS
        # -------------------------------------
        overweight = weights > W_use
        # strong negative penalty
        raw_penalized = np.where(overweight, -1e18, raw)

        # Apply city mask (non-tours stay zero)
        fit = raw_penalized * mask_city

        # -------------------------------------
        # Shift to positive (GA requirement)
        # -------------------------------------
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
        R=None,
        beta=None,
        W=None,
        alpha=None,
        **kw
    ):
        # accepts R, W, alpha, beta, **kw
        profits     = np.asarray(metric_values["profits"], dtype=np.float64)
        times       = np.asarray(metric_values["times"], dtype=np.float64)
        # we also use weights here for overweight penalty
        weights     = np.asarray(metric_values["weights"], dtype=np.float64)
        number_city = np.asarray(metric_values["number_city"], dtype=np.float64)

        # number_obj is optional; if metrics don't provide it,
        # we just use a factor of 1.0 for everyone.
        if "number_obj" in metric_values:
            number_obj = np.asarray(metric_values["number_obj"], dtype=np.float64)
        else:
            number_obj = np.ones_like(profits, dtype=np.float64)

        # 1) mask for full tours (same idea as project-1 __cityBinaryTSP)
        mask_city = (number_city >= self.GENOME_LENGTH).astype(np.float64)

        # 2) determine R and capacity Wmax using the helpers
        R_use = self._resolve_R(R)
        Wmax  = self._resolve_W(W)

        # 3) overweight penalty like in project-1 Fitness.fitnessF1scoreTTP
        overweight = np.maximum(0.0, weights - Wmax)

        # beta from argument or configs
        if beta is None:
            beta = self.__configs.get("beta", 1.0)
        beta = float(beta)

        penalty = beta * overweight

        # 4) base F1-like term, similar to:
        #       2 * profit / (profit + R * time + penalty)
        #    multiplied by:
        #       mask_city  (only full tours)
        #       number_obj (optional extra factor from metrics)
        raw = mask_city * number_obj * (
            2.0 * profits / (profits + R_use * times + penalty + 1e-9)
        )

        # 5) shift to positive (GA needs > 0 fitness)
        min_raw = raw.min()
        if min_raw <= 0:
            raw = raw - min_raw + 1e-6

        # 6) selective pressure exponent alpha
        if alpha is None:
            alpha = self.__configs.get("alpha", 2.0)
        alpha = float(alpha)

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
