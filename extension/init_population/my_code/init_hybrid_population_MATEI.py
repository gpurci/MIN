#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *
from extension.local_search_algorithms.kp_greedy import TTPKPLocalSearch
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.tabu_hybrid_search import TabuHybridSearch
from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.vnd import VND


class InitPopulationHybrid(RootGA):
    """
    Extension: Greedy neighbor-based TTP initial population (TTP_hybrid),
    self-contained version (NO dependency on Metrics).
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

        # Inject a TwoOpt operator for route refinement
        self.two_opt_operator = TwoOpt("two_opt", dataset)


        self.__fn = self.initPopulationHybrid


    def __str__(self):
        return f"InitPopulationHybrid(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""InitPopulationHybrid:
    metoda: 'TTP_vecin'; config:
        lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936, seed\n""")

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self.__configs)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if self.GENOME_LENGTH and not hasattr(self, "_all_cities"):
            self._all_cities = np.arange(self.GENOME_LENGTH, dtype=np.int32)

    def initPopulationHybrid(self, size, genoms=None,
                             lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936,
                             seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Precompute local search operators from your own classes
        two_opt_simple = self.two_opt_operator.twoOptSimple

        or_opt = OrOpt("or_opt_restrict", self.dataset)
        or_opt.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        tabu2 = TabuHybridSearch("hybrid_2opt", self.dataset)
        tabu2.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        vnd = VND(self.dataset)
        vnd.setParameters(GENOME_LENGTH=self.GENOME_LENGTH)

        kp_ls = TTPKPLocalSearch(self.dataset)

        seen_routes = set()
        count = 0
        N = size

        while count < size:
            # Randomize starting city to increase diversity
            start_city = np.random.randint(0, self.GENOME_LENGTH)

            # Small λ diversification (±20%)
            lam = lambda_time * np.random.uniform(0.8, 1.2)

            # 1) Build a greedy route
            r, kp = self._constructGreedyRoute(
                start_city, lam, vmax, vmin, Wmax)

            choice = np.random.rand()

            # -------- MIXING STRATEGY --------
            if choice < 0.30:
                # 30%: greedy + simple 2opt (cheap & diverse)
                r = two_opt_simple(r)

            elif choice < 0.55:
                # next 25%: greedy + OrOptRestrict (medium LS)
                r = or_opt(None, None, r)

            elif choice < 0.75:
                # next 20%: greedy + tabu hybrid2opt
                r = tabu2(None, None, r)

            elif choice < 0.90:
                # next 15%: the strongest — full VND
                r = vnd(None, None, r)

            else:
                # last 10%: random shuffle + VND for pure diversity
                r = np.random.permutation(self.GENOME_LENGTH)
                r = vnd(None, None, r)

            # refine knapsack using your KPGreedyImprove
            offspring = {"tsp": r, "kp": kp}
            offspring2 = kp_ls(None, None, offspring)
            new_kp = offspring2["kp"]

            # avoid exact duplicate routes
            tup = tuple(r)
            if tup in seen_routes:
                continue
            seen_routes.add(tup)

            # final individual
            genoms.add(tsp=r, kp=new_kp)
            count += 1

        genoms.save()
        print("Hybrid mixed TTP population =", genoms.shape)

    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        return v_max - (v_max - v_min) * (Wcur / float(Wmax))

    def _route_distance(self, route):
        d = self.distance[route[:-1], route[1:]].sum()
        d += self.distance[route[-1], route[0]]
        return d

    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
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

        # --- preallocate scratch buffers once ---
        dist_all = np.empty(n, dtype=np.float64)  # distances from cur
        score_all = np.empty(n, dtype=np.float64)  # adjusted score
        feas_all = np.empty(n, dtype=bool)  # feasibility mask
        score_mask = np.empty(n, dtype=np.float64)  # masked score for selection
        dist_mask = np.empty(n, dtype=np.float64)  # masked dist for fallback

        NEG_INF = -1e18
        POS_INF = 1e18

        for t in range(1, n):
            v_cur = self.computeSpeedTTP(Wcur, vmax, vmin, Wmax)
            inv_v = 1.0 / v_cur

            # dist_all = D[cur]  (copy into buffer to allow in-place masking later)
            dist_all[:] = D[cur]

            # score_all = max(0, P - lambda_time * dist_all / v_cur)
            # do it in-place:
            # score_all = dist_all * inv_v
            score_all[:] = dist_all
            score_all *= inv_v  # now score_all = time_all
            score_all *= lambda_time  # score_all = lambda_time * time_all
            score_all[:] = P - score_all  # score_all = P - lambda_time*time_all
            np.maximum(score_all, 0.0, out=score_all)

            # feas_all = (Wcur + W) <= Wmax   (in-place)
            # This allocates a temp from (Wcur + W) anyway, but it's one vector:
            feas_all[:] = (Wcur + W) <= Wmax

            # score_all *= feas_all   (zero infeasible)
            score_all *= feas_all

            # score_mask = score_all, but visited -> NEG_INF
            score_mask[:] = score_all
            score_mask[~unvisited] = NEG_INF

            if np.all(score_mask <= 0):
                # fallback nearest neighbor among unvisited
                dist_mask[:] = dist_all
                dist_mask[~unvisited] = POS_INF
                j = int(np.argmin(dist_mask))
            else:
                k = 5 if n >= 5 else n
                top_idx = np.argpartition(score_mask, -k)[-k:]
                # keep only valid
                top_idx = top_idx[(unvisited[top_idx]) & (score_mask[top_idx] > 0)]
                if top_idx.size == 0:
                    dist_mask[:] = dist_all
                    dist_mask[~unvisited] = POS_INF
                    j = int(np.argmin(dist_mask))
                else:
                    j = int(np.random.choice(top_idx))

            path[t] = j
            unvisited[j] = False

            # item decision (same logic)
            gain = P[j] - lambda_time * D[cur, j] * inv_v
            if gain > 0 and (Wcur + W[j] <= Wmax):
                Wcur += W[j]
                kp[j] = 1

            cur = j

        return path, kp

