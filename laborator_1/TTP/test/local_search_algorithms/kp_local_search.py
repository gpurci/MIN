
#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class TTPKPLocalSearch(RootGA):
    """
    Genome-level KP greedy improvement (TTP-aware)

    Correct GA interface:
        __call__(parent1, parent2, offspring)

    offspring is:
        {"tsp": array(n), "kp": array(n)}
    """

    def __init__(
        self,
        dataset,
        mode="ada_linear",
        v_min=0.1,
        v_max=1.0,
        W=25936,
        alpha=0.01,
        max_passes=2,
        seed=None,
    ):
        super().__init__()

        self.dataset = dataset
        self.distance = dataset["distance"]
        self.profit = dataset["item_profit"]
        self.weight = dataset["item_weight"]

        self.mode = mode
        self.v_min = v_min
        self.v_max = v_max
        self.Wmax = W
        self.alpha = alpha

        self.max_passes = max_passes
        self.seed = seed

    # ------------------------------------------------------
    # Allow GA to override parameters
    # ------------------------------------------------------
    def setParameters(self, **kw):
        super().setParameters(**kw)
        self.v_min = kw.get("v_min", self.v_min)
        self.v_max = kw.get("v_max", self.v_max)
        self.Wmax = kw.get("W", self.Wmax)
        self.alpha = kw.get("alpha", self.alpha)

    # ------------------------------------------------------
    # Speed function (TTP linear model)
    # ------------------------------------------------------
    def _speed(self, Wcur):
        if self.mode == "ada_linear":
            v = self.v_max - self.v_min * ((Wcur / self.Wmax) - 1.0)
        else:
            v = self.v_max - (self.v_max - self.v_min) * (Wcur / float(self.Wmax))
        return max(self.v_min, min(self.v_max, v))

    # ------------------------------------------------------
    # Simulation of TTP route
    # ------------------------------------------------------
    def _simulate(self, tsp, kp):
        n = len(tsp)
        visit_time = np.zeros(n, dtype=np.float64)
        visit_weight = np.zeros(n, dtype=np.float64)

        Wcur = 0.0
        Tcur = 0.0

        for i in range(n - 1):
            city = tsp[i]
            visit_time[city] = Tcur
            visit_weight[city] = Wcur

            if kp[city] == 1:
                p = self.profit[city]
                w = self.weight[city]

                if self.mode == "ada_linear":
                    contrib = max(0.0, (p * p) / (w + 1e-7) - self.alpha * Tcur)
                else:
                    contrib = max(0.0, p - self.alpha * Tcur)

                Wcur += w

            v = self._speed(Wcur)
            Tcur += self.distance[city, tsp[i + 1]] / v

        # return to depot
        v = self._speed(Wcur)
        Tcur += self.distance[tsp[-1], tsp[0]] / v

        return visit_time, visit_weight, Wcur, Tcur

    # ------------------------------------------------------
    # Main call — GA interface
    # ------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **kw):
        """
        offspring: {"tsp":..., "kp":...}
        Returns: same structure with improved kp
        """

        out = offspring.copy()
        tsp = out["tsp"]
        kp = out["kp"].copy()

        if self.seed is not None:
            np.random.seed(self.seed)

        # ----- Iterative improvement -----
        for _ in range(self.max_passes):
            visit_time, visit_weight, Wcur, Tcur = self._simulate(tsp, kp)

            # ADD PASS
            remaining = self.Wmax - Wcur
            candidates = []

            for city in tsp:
                if kp[city] == 1:
                    continue

                w = self.weight[city]
                if w > remaining:
                    continue

                p = self.profit[city]
                t_at = visit_time[city]

                if self.mode == "ada_linear":
                    gain = max(0.0, (p * p) / (w + 1e-7) - self.alpha * t_at)
                else:
                    gain = max(0.0, p - self.alpha * t_at)

                if gain > 0:
                    candidates.append((gain / (w + 1e-7), city, gain, w))

            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, city, gain, w in candidates:
                if Wcur + w <= self.Wmax:
                    kp[city] = 1
                    Wcur += w

            # REMOVE PASS
            visit_time, visit_weight, Wcur, Tcur = self._simulate(tsp, kp)
            removed = False

            for city in tsp:
                if kp[city] == 0:
                    continue

                w = self.weight[city]
                p = self.profit[city]
                t_at = visit_time[city]

                if self.mode == "ada_linear":
                    contrib = max(0.0, (p * p) / (w + 1e-7) - self.alpha * t_at)
                else:
                    contrib = max(0.0, p - self.alpha * t_at)

                if contrib <= 0:
                    kp[city] = 0
                    removed = True

            if not removed:
                break

        # ALWAYS enforce depot rule
        if kp.shape[0] > 0:
            kp[0] = 0

        out["kp"] = kp
        return out
