#!/usr/bin/python
import numpy as np
from root_GA import *
from genoms import *


class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.

    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Metoda '__config_fn' selecteaza functia de initializare.
    Metoda '__call__' aplica functia selectata.
    """

    def __init__(self, method, metrics, genoms, **kw):
        super().__init__()
        self.metrics = metrics          # obiect Metrics (TTP dataset)
        self.__genoms = genoms
        self.__configs = kw
        print("DEBUG InitPopulation: method =", method, "configs =", kw)
        self.__setMethods(method)

    def __str__(self):
        info = f"InitPopulation: method: {self.__method} configs: {self.__configs}"
        return info

    def __call__(self, size):
        """apel direct: obiect(config)(size)"""
        return self.fn(size, **self.__configs)

    # ------------------------------------------------------------------
    def __unpack_method(self, method):
        """Selecteaza metoda de initializare."""
        fn = self.initPopulationAbstract
        if method is not None:
            if method == "TTP_vecin":
                fn = self.initPopulationTTP
            elif method == "TSP_rand":
                fn = self.initPopulationsTSPRand
            elif method == "TTP_rand":
                fn = self.initPopulationsTTPRand
            elif method == "TTP_rand_mix":
                fn = self.initPopulationTTP_rand_mix

        return fn

    def help(self):
        return (
            "InitPopulation: "
            "metoda: 'TTP_vecin'; config: -> lambda_time, vmax, vmin, Wmax, seed;\n"
            "metoda: 'TTP_rand'; config: None;\n"
            "metoda: 'TSP_rand'; config: None;\n"
        )

    # ------------------------------------------------------------------
    def __setMethods(self, method):
        self.__method = method
        self.fn = self.__unpack_method(method)

    # ------------------------------------------------------------------
    def initPopulationAbstract(self, size):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru functia de 'InitPopulation': "
            f"config '{self.__configs}'"
        )

    # ================================================================
    #                 RANDOM TSP INITIALIZATION
    # ================================================================
    def initPopulationsTSPRand(self, population_size=-1):
        """Initializare TSP random."""
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)

        for _ in range(population_size):
            self.__genoms.add(tsp=np.random.permutation(individ))

        self.__genoms.save()
        print("population", self.__genoms.shape)

    # ================================================================
    #                 RANDOM TTP INITIALIZATION
    # ================================================================
    def initPopulationsTTPRand(self, population_size=-1):
        """Initializare TTP random (TSP random + KP random)."""
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)

        for _ in range(population_size):
            kp_individ = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            self.__genoms.add(
                tsp=np.random.permutation(tsp_individ),
                kp=kp_individ
            )

        self.__genoms.save()
        print("population", self.__genoms.shape)

    # ================================================================
    #                 GREEDY TTP INITIALIZATION
    # ================================================================
    def initPopulationTTP_rand_mix(self, size, ratio=0.1):
        """
        Mixed initialization:
        ratio portion = greedy TTP_vecin (with 2-opt)
        rest          = random TTP_rand

        ✨ This is only called once → 2-opt here is cheap.
        """

        n_vecin = int(size * ratio)
        n_rand = size - n_vecin

        vecin_list = []
        attempts = 0
        max_attempts = n_vecin * 50

        # load TTP dataset once
        self._loadTTPdataset()

        # ---------- produce greedy vecin individuals ----------
        seen = set()
        while len(vecin_list) < n_vecin and attempts < max_attempts:
            attempts += 1

            # build route
            route, kp = self._constructGreedyRoute(
                start=0,
                lambda_time=0.1,
                vmax=1.0,
                vmin=0.1,
                Wmax=25936
            )

            # 🔄 CHANGED: cheaper 2-opt (max_iter=50)
            route = self.local_search_2opt(route, self.distance, max_iter=50)

            key = tuple(route)
            if key in seen:
                continue

            seen.add(key)
            vecin_list.append({"tsp": route, "kp": kp})

        if len(vecin_list) < n_vecin:
            print(f"[WARN] Only {len(vecin_list)}/{n_vecin} unique greedy individuals generated.")

        # ---------- produce random individuals ----------
        rand_list = []
        for _ in range(n_rand):
            tsp = np.random.permutation(np.arange(self.GENOME_LENGTH))
            kp = np.random.randint(0, 2, self.GENOME_LENGTH)
            rand_list.append({"tsp": tsp, "kp": kp})

        # ---------- build final population ----------
        self.__genoms.clear()

        for indiv in vecin_list + rand_list:
            self.__genoms.add(**indiv)

        self.__genoms.save()

        print(f"population initialized (mixed): {self.__genoms.shape}")

    # ================================================================
    #                         TTP DATASET I/O
    # ================================================================
    def _loadTTPdataset(self):
        dataset = self.metrics.getDataset()
        self.coords = dataset["coords"]
        self.distance = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

    # ================================================================
    #               GREEDY ROUTE CONSTRUCTION FOR TTP
    # ================================================================
    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        """
        Greedy TTP: scor = profit - λ * time.
        1 item per city (TTP standard).
        """
        start = 0
        visited = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited[start] = True

        path = [start]
        kp = np.zeros(self.GENOME_LENGTH, dtype=np.int32)

        cur = start
        Wcur = 0.0

        for _ in range(self.GENOME_LENGTH - 1):

            cand = np.where(~visited)[0]
            if cand.size == 0:
                break

            v_cur = vmax - (vmax - vmin) * (Wcur / Wmax)
            v_cur = max(vmin, v_cur)

            dist = self.distance[cur, cand]
            time_to_candidate = dist / v_cur
            profit_raw = self.item_profit[cand].astype(float)

            profit_if_take = np.maximum(
                0.0,
                profit_raw - lambda_time * time_to_candidate
            )

            can_take = (Wcur + self.item_weight[cand]) <= Wmax
            score = profit_if_take * can_take

            # fallback: nearest
            if np.all(score == 0):
                j = cand[np.argmin(dist)]
            else:
                order = np.argsort(score)
                top_k = min(5, len(order))
                j = np.random.choice(cand[order[-top_k:]])

            path.append(j)
            visited[j] = True

            # picking decision
            w_j = float(self.item_weight[j])
            p_j = float(self.item_profit[j])

            if (Wcur + w_j) <= Wmax:
                tj = self.distance[cur, j] / v_cur
                gain = p_j - lambda_time * tj
                if gain > 0:
                    Wcur += w_j
                    kp[j] = 1

            cur = j

        return np.array(path, dtype=np.int32), kp

    # ================================================================
    #                    FULL LOCAL-SEARCH 2-OPT
    # ================================================================
    def local_search_2opt(self, route, distance, max_iter=200):
        """
        Full local-search 2-opt (best-improvement or first-improvement).
        Repeats until no improvement is found or max_iter reached.

        route: 1D numpy array, permutation of cities
        distance: precomputed distance matrix (NxN)
        max_iter: safety stop
        """
        n = len(route)
        best = route.copy()

        def dist(i, j):
            return distance[i, j]

        def route_length(r):
            return dist(r[-1], r[0]) + np.sum(distance[r[:-1], r[1:]])

        best_len = route_length(best)
        improved = True
        it = 0

        while improved and it < max_iter:
            improved = False
            it += 1

            for i in range(1, n - 2):
                a, b = best[i - 1], best[i]
                for k in range(i + 1, n - 1):
                    c, d = best[k], best[k + 1]

                    # compute gain without rebuilding whole route
                    gain = dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d)

                    if gain < -1e-12:
                        # apply 2-opt inversion
                        best[i:k+1] = best[i:k+1][::-1]
                        best_len += gain
                        improved = True
                        break

                if improved:
                    break

        return best
