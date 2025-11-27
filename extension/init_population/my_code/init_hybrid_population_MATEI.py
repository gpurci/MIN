#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

from extension.local_search_algorithms.kp_local_search import TTPKPLocalSearch
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.tabu_hybrid_search import TabuHybridSearch
from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.vnd import VND


class InitPopulationHybrid(RootGA):
    """
    Hybrid TTP initial population generator (TTP_hybrid).
    Uses greedy constructive heuristic for TTP + light local-search refinement.

    *** IMPORTANT ***
    This version does NOT use TTP-VND (too heavy for initialization).
    Only lightweight route improvements are applied to preserve diversity.
    """

    def __init__(self, method="TTP_hybrid", dataset=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method

        if dataset is None:
            raise ValueError("InitPopulationHybrid requires dataset=<TTP dataset dict>")

        self.dataset = dataset
        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        # Cheap per-route 2-opt operator
        self.two_opt_operator = TwoOpt("two_opt_LS", dataset)

        self.__fn = self.initPopulationHybrid

    def __str__(self):
        return f"InitPopulationHybrid(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""InitPopulationHybrid:
    metoda: 'TTP_hybrid'; config:
        lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936, seed\n""")

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self.__configs)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if self.GENOME_LENGTH and not hasattr(self, "_all_cities"):
            self._all_cities = np.arange(self.GENOME_LENGTH, dtype=np.int32)

    # ==================================================================
    #                      HYBRID INITIAL POPULATION
    # ==================================================================
    def initPopulationHybrid(
        self, size, genoms=None, lambda_time=0.1,
        vmax=1.0, vmin=0.1, Wmax=25936, seed=None
    ):

        if seed is not None:
            np.random.seed(seed)

        # Precompute LS operators (light-weight)
        two_opt_simple = self.two_opt_operator.twoOptLS

        or_opt = OrOpt("or_opt_restrict", self.dataset)
        or_opt.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        tabu2 = TabuHybridSearch("hybrid_2opt", self.dataset)
        tabu2.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        # Simple VND (TSP-only)
        vnd = VND(self.dataset)
        vnd.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        # Lightweight KP local search
        kp_ls = TTPKPLocalSearch(self.dataset)

        seen_routes = set()
        count = 0

        # ------------------------------------------------------------------
        # Limit expensive KP local search to a small budget
        # For example: 10% of the population, at least 10 individuals
        # This has a *huge* impact on init time.
        kpls_budget = max(10, size // 10)
        kpls_used = 0

        while count < size:

            # ------------------------------------------------------------------
            # (1) Build greedy TTP route + KP
            # ------------------------------------------------------------------
            start_city = np.random.randint(0, self.GENOME_LENGTH)
            lam = lambda_time * np.random.uniform(0.8, 1.2)

            r, kp = self._constructGreedyRoute(start_city, lam, vmax, vmin, Wmax)

            # ------------------------------------------------------------------
            # (2) Light post-processing (cheap)
            # ------------------------------------------------------------------
            choice = np.random.rand()

            if choice < 0.30:
                r = two_opt_simple(r)     # cheap 2-opt

            elif choice < 0.55:
                r = or_opt(None, None, r) # medium LS

            elif choice < 0.75:
                r = tabu2(None, None, r)  # tabu 2-opt

            elif choice < 0.90:
                r = vnd(None, None, r)    # simple VND

            else:
                # Forced diversity
                r = np.random.permutation(self.GENOME_LENGTH)
                r = vnd(None, None, r)

            # ------------------------------------------------------------------
            # (3) Repair KP
            # ------------------------------------------------------------------
            offspring = {"tsp": r, "kp": kp}

            # Only run expensive KP LS for a limited number of individuals
            if kpls_used < kpls_budget:
                offspring2 = kp_ls(None, None, offspring)
                new_kp = offspring2["kp"]
                kpls_used += 1
            else:
                # After budget is exhausted, keep the greedy KP as-is
                new_kp = kp

            # ------------------------------------------------------------------
            # Avoid exact duplicates
            # ------------------------------------------------------------------
            key = hash(r.tobytes())
            if key in seen_routes:
                continue
            seen_routes.add(key)

            # Add final hybrid solution
            genoms.add(tsp=r, kp=new_kp)
            count += 1


        genoms.save()
        print("Hybrid mixed TTP population =", genoms.shape)

    # ==================================================================
    # Utility methods
    # ==================================================================
    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        return v_max - (v_max - v_min) * (Wcur / float(Wmax))

    def _route_distance(self, route):
        d = self.distance[route[:-1], route[1:]].sum()
        d += self.distance[route[-1], route[0]]
        return d

    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        """
        Core TTP greedy heuristic:
            - choose next city by combined profit/time scoring
            - choose item if beneficial and feasible
        """
        n = self.GENOME_LENGTH
        D = self.distance
        P = self.item_profit
        W = self.item_weight

        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False

        path = np.empty(n, dtype=np.int32)
        kp = np.zeros(n, dtype=np.int32)

        path[0] = start
        cur = start
        Wcur = 0.0

        dist_all = np.empty(n, dtype=np.float64)
        score_all = np.empty(n, dtype=np.float64)
        feas_all  = np.empty(n, dtype=bool)
        score_mask = np.empty(n, dtype=np.float64)
        dist_mask  = np.empty(n, dtype=np.float64)

        NEG_INF = -1e18
        POS_INF = 1e18

        for t in range(1, n):
            v_cur = self.computeSpeedTTP(Wcur, vmax, vmin, Wmax)
            inv_v = 1.0 / v_cur

            dist_all[:] = D[cur]

            score_all[:] = dist_all
            score_all *= inv_v
            score_all *= lambda_time
            score_all[:] = P - score_all
            np.maximum(score_all, 0.0, out=score_all)

            feas_all[:] = (Wcur + W) <= Wmax
            score_all *= feas_all

            score_mask[:] = score_all
            score_mask[~unvisited] = NEG_INF

            if np.all(score_mask <= 0):
                dist_mask[:] = dist_all
                dist_mask[~unvisited] = POS_INF
                j = int(np.argmin(dist_mask))
            else:
                k = 5 if n >= 5 else n
                top_idx = np.argpartition(score_mask, -k)[-k:]
                top_idx = top_idx[(unvisited[top_idx]) & (score_mask[top_idx] > 0)]
                if top_idx.size == 0:
                    dist_mask[:] = dist_all
                    dist_mask[~unvisited] = POS_INF
                    j = int(np.argmin(dist_mask))
                else:
                    j = int(np.random.choice(top_idx))

            path[t] = j
            unvisited[j] = False

            gain = P[j] - lambda_time * D[cur, j] * inv_v
            if gain > 0 and (Wcur + W[j] <= Wmax):
                Wcur += W[j]
                kp[j] = 1

            cur = j

        return path, kp
