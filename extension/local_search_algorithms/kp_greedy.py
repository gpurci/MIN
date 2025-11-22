#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class KPGreedyImprove(RootGA):
    """
    KP-aware local search for TTP.
    Improves the knapsack chromosome given a fixed TSP route.

    Idea:
      - simulate traveling along tsp
      - estimate profit contribution of each city item at its visit time
      - greedily add high-value items while feasible
      - remove items with non-positive contribution
      - repeat a couple passes

    Defaults match your metrics config:
        mode="ada_linear"
        v_min=0.1, v_max=1.0, W=25936, alpha=0.01

    Usage:
        kp_ls = KPGreedyImprove(dataset, mode="ada_linear")
        kp2 = kp_ls(tsp, kp)
    """

    def __init__(self, dataset, mode="ada_linear",
                 v_min=0.1, v_max=1.0, W=25936, alpha=0.01,
                 max_passes=2, seed=None):
        super().__init__()
        self.dataset = dataset
        self.distance = dataset["distance"]
        self.profit = dataset["item_profit"]
        self.weight = dataset["item_weight"]

        self.mode = mode
        self.v_min = v_min
        self.v_max = v_max
        self.Wmax  = W
        self.alpha = alpha

        self.max_passes = max_passes
        self.seed = seed

    def setParameters(self, **kw):
        """
        Let GA overwrite parameters if it provides them.
        """
        super().setParameters(**kw)
        self.v_min = kw.get("v_min", self.v_min)
        self.v_max = kw.get("v_max", self.v_max)
        self.Wmax  = kw.get("W", self.Wmax)
        self.alpha = kw.get("alpha", self.alpha)

    # ---------- core TTP simulation ----------
    def _speed(self, Wcur):
        # match your ada_linear speed in MetricsTTP
        if self.mode == "ada_linear":
            v = self.v_max - self.v_min * ((Wcur / self.Wmax) - 1.0)
        else:
            v = self.v_max - (self.v_max - self.v_min) * (Wcur / float(self.Wmax))
        return max(self.v_min, min(self.v_max, v))

    def _simulate(self, tsp, kp):
        """
        Simulate route to get:
          - visit_time[city]  (time when city is visited)
          - visit_weight[city] (current bag weight at city)
          - total_score (ttp profit proxy)
          - total_weight, total_time
        """
        n = len(tsp)
        visit_time   = np.zeros(n, dtype=np.float64)
        visit_weight = np.zeros(n, dtype=np.float64)

        Wcur = 0.0
        Tcur = 0.0
        score = 0.0

        for i in range(n - 1):
            city = tsp[i]
            take = kp[city]

            visit_time[city] = Tcur
            visit_weight[city] = Wcur

            if take:
                p = self.profit[city]
                w = self.weight[city]

                if self.mode == "ada_linear":
                    contrib = max(0.0, (p * p) / (w + 1e-7) - self.alpha * Tcur)
                else:
                    contrib = max(0.0, p - self.alpha * Tcur)

                score += contrib
                Wcur += w

            v = self._speed(Wcur)
            Tcur += self.distance[city, tsp[i + 1]] / v

        # return to start
        v = self._speed(Wcur)
        Tcur += self.distance[tsp[-1], tsp[0]] / v

        return visit_time, visit_weight, score, Wcur, Tcur

    # ---------- greedy improvement ----------
    def __call__(self, tsp, kp):
        if self.seed is not None:
            np.random.seed(self.seed)

        kp2 = kp.copy()

        for _ in range(self.max_passes):
            visit_time, visit_weight, score, Wcur, Tcur = self._simulate(tsp, kp2)

            # --- ADD PASS (greedy by adjusted gain/weight) ---
            remaining = self.Wmax - Wcur
            candidates = []

            for city in tsp:
                if kp2[city] == 1:
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
                    kp2[city] = 1
                    Wcur += w

            # --- REMOVE PASS (drop useless items) ---
            visit_time, visit_weight, score, Wcur, Tcur = self._simulate(tsp, kp2)

            removed_any = False
            for city in tsp:
                if kp2[city] == 0:
                    continue
                w = self.weight[city]
                p = self.profit[city]
                t_at = visit_time[city]

                if self.mode == "ada_linear":
                    contrib = max(0.0, (p * p) / (w + 1e-7) - self.alpha * t_at)
                else:
                    contrib = max(0.0, p - self.alpha * t_at)

                if contrib <= 0:
                    kp2[city] = 0
                    removed_any = True

            if not removed_any:
                break

        return kp2


class TTPKPLocalSearch(RootGA):
    """
    Genome-level local search:
      offspring is a TTP genome with fields ["tsp","kp"].
      We improve only kp using KPGreedyImprove.

    Use this as a GA operator when method="genome"
    or inside an elite search chain.

    Usage:
        kp_ls = TTPKPLocalSearch(dataset, mode="ada_linear")
        genome2 = kp_ls(None, None, genome)
    """

    def __init__(self, dataset, **configs):
        super().__init__()
        self.kp_search = KPGreedyImprove(dataset, **configs)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        self.kp_search.setParameters(**kw)

    def __call__(self, parent1, parent2, offspring):
        out = offspring.copy()
        tsp = out["tsp"]
        kp  = out["kp"]
        out["kp"] = self.kp_search(tsp, kp)
        return out
