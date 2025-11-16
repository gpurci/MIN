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
    def initPopulationTTP(self, size, lambda_time=0.1, vmax=1.0,
                          vmin=0.1, Wmax=25936, seed=None):
        """
        Genereaza size indivizi folosind o euristica greedy TTP:
        - fiecare individ incepe din orasul 0
        - scor = profit - λ * timp
        - dupa ce rute se construiesc → aplicam o singură trecere 2-opt
        """

        if seed is not None:
            np.random.seed(seed)

        self._loadTTPdataset()
        seen = set()
        count = 0

        while count < size:
            start_city = 0

            route, kp = self._constructGreedyRoute(
                start_city, lambda_time, vmax, vmin, Wmax
            )

            # aplicam 2-opt
            route = self._twoOpt(route)

            # evitam duplicatele TSP
            key = tuple(route)
            if key in seen:
                continue
            seen.add(key)

            # salvam in Genoms
            self.__genoms.add(tsp=route, kp=kp)
            count += 1

        self.__genoms.save()
        print("population initialized:", self.__genoms.shape)

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
    #                         ONE-PASS 2-OPT
    # ================================================================
    def _twoOpt(self, route):
        """
        Single-pass 2-opt:
        Găsește prima îmbunătățire și se oprește.
        """
        n = len(route) - 1

        def route_length(r):
            return self.distance[r[:-1], r[1:]].sum()

        best = route.copy()
        best_len = route_length(best)

        for i in range(1, n - 1):
            for k in range(i + 1, n):
                new = best.copy()
                new[i:k+1] = new[i:k+1][::-1]
                new_len = route_length(new)

                if new_len < best_len - 1e-9:
                    return new  # prima imbunatatire

        return best
