#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

from extension.local_search_algorithms.kp_local_search import TTPKPLocalSearch
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.tabu_hybrid_search import TabuHybridSearch
from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.vnd import VND
from extension.local_search_algorithms.ttp_vnd import TTPVNDLocalSearch


class InitPopulationHybrid(RootGA):
    """
    Hybrid TTP initial population generator (TTP_hybrid).
    Uses greedy constructive heuristic for TTP + light local-search refinement.

    NEW:
      * Optional TTP-aware VND on a small fraction of the initial population
        (use_ttp_vnd_init=True, init_vnd_frac ~ 0.10–0.20).
    """

    def __init__(self, method="TTP_hybrid", dataset=None, **configs):
        super().__init__()
        # IMPORTANT: copy configs so we can safely pop stuff
        self.__configs = dict(configs)
        self.__method = method

        if dataset is None:
            raise ValueError("InitPopulationHybrid requires dataset=<TTP dataset dict>")

        self.dataset = dataset
        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        # Cheap per-route 2-opt operator
        self.two_opt_operator = TwoOpt("two_opt_LS", dataset)

        # ---------- NEW: TTP-aware VND init options ----------
        # Pop them out so they are NOT passed to initPopulationHybrid(...)
        self.use_ttp_vnd_init = bool(self.__configs.pop("use_ttp_vnd_init", False))
        self.init_vnd_frac    = float(self.__configs.pop("init_vnd_frac", 0.15))
        self.init_vnd_rounds  = int(self.__configs.pop("init_vnd_rounds", 2))

        if self.use_ttp_vnd_init:
            self.ttp_vnd_init = TTPVNDLocalSearch(
                dataset=dataset,
                v_max=dataset["v_max"],
                v_min=dataset["v_min"],
                W=dataset["W"],
                R=dataset["R"],
                max_rounds=self.init_vnd_rounds,
                use_kp_ls=True,
                use_tsp_ls=True,
            )
        else:
            self.ttp_vnd_init = None

        self.__fn = self.initPopulationHybrid

    def __str__(self):
        return (
            f"InitPopulationHybrid(method={self.__method}, "
            f"use_ttp_vnd_init={self.use_ttp_vnd_init}, "
            f"init_vnd_frac={self.init_vnd_frac}, "
            f"init_vnd_rounds={self.init_vnd_rounds}, "
            f"configs={self.__configs})"
        )

    def help(self):
        print("""InitPopulationHybrid:
    method: 'TTP_hybrid'; config:
        lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936, seed
        use_ttp_vnd_init (bool)
        init_vnd_frac (float, e.g. 0.15)
        init_vnd_rounds (int, e.g. 2)
""")

    # ==================================================================
    # __call__ supports 2 modes:
    #   (1) GA InitPopulation wrapper: extern_fn(size) -> dict("tsp", "kp")
    #   (2) direct Genoms filling: extern_fn(size, genoms=genoms) -> None
    # ==================================================================
    def __call__(self, size, genoms=None):
        if genoms is None:
            # Mode 1: return dict of arrays for Genoms.concatChromosomes(...)
            routes = []
            kps    = []

            class _Collector:
                def __init__(self, routes_list, kps_list):
                    self._routes = routes_list
                    self._kps    = kps_list
                    self.shape = (0, 0, 0)

                def add(self, tsp=None, kp=None, **_):
                    tsp = np.asarray(tsp, dtype=np.int32)
                    kp  = np.asarray(kp,  dtype=np.int32)
                    self._routes.append(tsp)
                    self._kps.append(kp)
                    self.shape = (len(self._routes), 2, tsp.shape[0])

                def save(self):
                    pass

            collector = _Collector(routes, kps)
            self.__fn(size, genoms=collector, **self.__configs)

            tsp_arr = np.stack(routes, axis=0)
            kp_arr  = np.stack(kps,    axis=0)
            return {"tsp": tsp_arr, "kp": kp_arr}

        # Mode 2: fill an existing Genoms object
        self.__fn(size, genoms=genoms, **self.__configs)

    def setParameters(self, **kw):
            super().setParameters(**kw)

            # existing bit
            if self.GENOME_LENGTH and not hasattr(self, "_all_cities"):
                self._all_cities = np.arange(self.GENOME_LENGTH, dtype=np.int32)

            # *** NEW: forward params into the TTP-VND init LS ***
            if getattr(self, "ttp_vnd_init", None) is not None:
                # This will pass GENOME_LENGTH, dataset, etc.
                self.ttp_vnd_init.setParameters(**kw)

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

        # Budget for KP-LS (already in your code)
        kpls_budget = max(10, size // 10)
        kpls_used = 0

        # ---------- NEW: budget for TTP-aware VND at init ----------
        if self.use_ttp_vnd_init and (self.ttp_vnd_init is not None):
            vnd_budget = max(5, int(size * self.init_vnd_frac))
        else:
            vnd_budget = 0
        vnd_used = 0

        while count < size:
            # (1) Build greedy TTP route + KP
            start_city = np.random.randint(0, self.GENOME_LENGTH)
            lam = lambda_time * np.random.uniform(0.8, 1.2)

            r, kp = self._constructGreedyRoute(start_city, lam, vmax, vmin, Wmax)

            # (2) Light TSP LS
            choice = np.random.rand()
            if choice < 0.30:
                r = two_opt_simple(r)
            elif choice < 0.55:
                r = or_opt(None, None, r)
            elif choice < 0.75:
                r = tabu2(None, None, r)
            elif choice < 0.90:
                r = vnd(None, None, r)
            else:
                # forced diversity
                r = np.random.permutation(self.GENOME_LENGTH)
                r = vnd(None, None, r)

            # (3) KP repair / LS
            offspring = {"tsp": r, "kp": kp}
            if kpls_used < kpls_budget:
                offspring2 = kp_ls(None, None, offspring)
                new_kp = offspring2["kp"]
                kpls_used += 1
            else:
                new_kp = kp

            # (4) NEW: TTP-aware VND on a small subset of individuals
            if (vnd_used < vnd_budget) and (self.ttp_vnd_init is not None):
                cand = {"tsp": r, "kp": new_kp}
                cand2 = self.ttp_vnd_init(None, None, cand)
                r = cand2["tsp"]
                new_kp = cand2["kp"]
                vnd_used += 1

            # Avoid duplicates
            key = hash(r.tobytes())
            if key in seen_routes:
                continue
            seen_routes.add(key)

            # Add final individual
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
