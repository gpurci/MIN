#!/usr/bin/python
import numpy as np
from root_GA import *
from genoms import *

class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, method, metrics, genoms, **kw):
        super().__init__()
        # metrics = obiectul Metrics — folosit pentru dataset
        self.metrics   = metrics
        self.__genoms  = genoms
        self.__configs = kw
        self.__setMethods(method)

    def __str__(self):
        info = """InitPopulation: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __call__(self, size):
        # apel direct: obiect(config)(size)
        return self.fn(size, **self.__configs)

    def __method_fn(self):
        # selecteaza metoda dupa care se aplica metrica
        self.fn = self.initPopulationAbstract
        if (self.__method is not None):
            if   (self.__method == "TTP_vecin"):
                self.fn = self.initPopulationTTP
            elif (self.__method == "TSP_rand"):
                self.fn = self.initPopulationsTSPRand
            elif (self.__method == "TTP_rand"):
                self.fn = self.initPopulationsTTPRand
        else:
            pass

    def help(self):
        info = """InitPopulation:
    metoda: 'TTP_vecin'; config: -> lambda_time, vmax, vmin, Wmax, seed;
    metoda: 'TTP_rand';  config: None;
    metoda: 'TSP_rand';  config: None;\n"""
        return info

    def __setMethods(self, method):
        self.__method = method
        self.__method_fn()

    def initPopulationAbstract(self, size):
        # default: nu exista implementare
        raise NameError("Lipseste metoda '{}' pentru functia de 'InitPopulation': config '{}'".format(self.__method, self.__configs))

    # initPopulationsTSPRand -------------------------------------
    def initPopulationsTSPRand(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga individ in genome
            self.__genoms.add(tsp=np.random.permutation(individ))
        # adauga indivizi in noua generatie
        self.__genoms.save()
        print("population {}".format(self.__genoms.shape))
    # initPopulationsTSPRand =====================================

    # initPopulationRand -------------------------------------
    def initPopulationsTTPRand(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        routes = []
        kps = []
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga tsp_individ in genome
            tsp = np.random.permutation(tsp_individ)
            kp = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            routes.append(tsp)
            kps.append(kp)
        # adauga indivizi in noua generatie
        routes = np.array(routes, dtype=np.int32)
        kps = np.array(kps, dtype=np.int32)
        print("population {}".format(self.__genoms.shape))
        return routes, kps

    # initPopulationRand =====================================
    # initPopulationMatei -------------------------------------
    def initPopulationTTP(self, size=2000, lambda_time=0.1,
                            vmax=1.0, vmin=0.1, Wmax=25936, seed=None):
        """
        Genereaza `size` indivizi folosind o euristica greedy TTP:
        - fiecare individ incepe dintr-un oras random
        - la fiecare pas alegem urmatorul oras dupa: profit - λ * timp_de_calatorie
        - dupa ce rute se construiesc → aplicam 2-opt simplu

    Generează populația inițială TTP.
    Pentru fiecare individ se alege un oraș de start random și se construiește ruta
    alegând la fiecare pas următorul oraș în funcție de un scor simplu:
        scor = profit - λ * timp_de_deplasare
    După construirea rutei se aplică o singură îmbunătățire 2-opt (+ eliminare duplicate).
    Returnează un array de rute valide (start == end).
        """

        if seed is not None:
            np.random.seed(seed)

        # load TTP dataset
        self._loadTTPdataset()

        seen = set()  # avoid duplicate TSP routes

        count = 0
        while count < size:

            # always start from city 0 (as required)
            start_city = 0

            # greedy TTP route + KP vector
            route, kp = self._constructGreedyRoute(
                start_city, lambda_time, vmax, vmin, Wmax
            )

            # apply one-pass 2-opt
            route = self._twoOpt(route)

            # avoid duplicates
            key = tuple(route)
            if key in seen:
                continue
            seen.add(key)

            # add to Genoms object
            self.__genoms.add(tsp=route, kp=kp)

            count += 1

        # finalize new generation
        self.__genoms.save()

        print("population initialized:", self.__genoms.shape)

    # HELPER — incarcare dataset TTP
    def _loadTTPdataset(self):
        dataset = self.metrics.getDataset()
        self.coords      = dataset["coords"]
        self.distance    = dataset["distance"]    # matrice NxN CEIL_2D
        self.item_profit = dataset["item_profit"] # vector de profit per oras
        self.item_weight = dataset["item_weight"] # vector de weight per oras

    # construieste o ruta greedy pornind dintr-un oras
    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        """
        Greedy TTP route construction:
        - always starts from city 0
        - builds both the TSP route and KP vector (0/1 item selection)
        - uses the Profit - λ * Time heuristic
        - selects 0 or 1 item per city (even if multiple exist in dataset)

        Returns:
            route: np.array of shape (GENOME_LENGTH+1)
            kp:    np.array of shape (GENOME_LENGTH)  (0/1 item pick)
        """

        # Force start from city 0 regardless of input
        start = 0

        visited = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited[start] = True

        path = [start]
        kp = np.zeros(self.GENOME_LENGTH, dtype=np.int32)

        cur = start
        Wcur = 0.0
        Tcur = 0.0

        # current knapsack weight
        Wcur = 0.0

        # one KP decision per city
        kp = np.zeros(self.GENOME_LENGTH, dtype=np.int32)

        # Build route over remaining cities
        for _ in range(self.GENOME_LENGTH - 1):

            cand = np.where(~visited)[0]

            # current speed based on weight
            v_cur = self.metrics.computeSpeedTTP(Wcur, vmax, vmin, Wmax)

            dist = self.distance[cur, cand]
            time_to_candidate = dist / v_cur

            # ---- profit for each candidate city ----
            # self.metrics.items contains: (city_index, weight, profit)
            profit_raw = np.zeros_like(cand, dtype=float)

            for i, c in enumerate(cand):
                # Sum profit of all items in that city
                profit_raw[i] = sum(
                    p for (city_id, w, p) in self.metrics.items if city_id == c
                )

            # profit after time penalty
            profit_if_take = np.maximum(0.0, profit_raw - lambda_time * time)

            # check if the *smallest* item from each city fits
            min_item_weight = np.zeros_like(cand, dtype=float)
            for i, c in enumerate(cand):
                # get the minimum weight item in that city
                items_c = [w for (city_id, w, p) in self.metrics.items if city_id == c]
                min_item_weight[i] = min(items_c) if items_c else 0.0

            can_take = (Wcur + min_item_weight) <= Wmax

            # greedy score = profit after penalty if feasible
            score = profit_if_take * can_take

            # pick from top-k (adds some randomness)
            order = np.argsort(score)
            top_k = min(5, len(order))
            choices = cand[order[-top_k:]]

            j = np.random.choice(choices)
            path.append(j)
            visited[j] = True

            # ===== Decide item picking for city j (0 or 1 item) =====
            # Compute exact gain for arrival at city j
            dist_j = self.distance[cur, j]
            time_j = dist_j / v_cur

            # candidate items in that city
            items_j = [(w, p) for (city_id, w, p) in self.metrics.items if city_id == j]

            if items_j:
                # select the SINGLE item with highest penalized gain
                best_gain = -1e9
                best_item_weight = None

                for (w, p) in items_j:
                    gain = p - lambda_time * time_j
                    if gain > best_gain and (Wcur + w) <= Wmax:
                        best_gain = gain
                        best_item_weight = w

                # take this one best item (if positive gain)
                if best_gain > 0 and best_item_weight is not None:
                    Wcur += best_item_weight
                    kp[j] = 1

            cur = j

        # Close route by returning to city 0
        path.append(path[0])

        return np.array(path, dtype=np.int32), kp

    # one-shot 2-opt improvement
    def _twoOpt(self, route):
        """
        single-pass 2-opt: testeaza O(N^2) swap-uri
        si se opreste la PRIMA imbunatatire gasita.
        """
        best = route.copy()
        best_dist = self.metrics.getIndividDistanceTTP(best, self.distance)
        n = len(route) - 1

        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new_route = best.copy()
                new_route[i:k] = best[k-1:i-1:-1]

                d = self.metrics.getIndividDistanceTTP(new_route, self.distance)
                if d < best_dist:
                    return new_route     # improvement found — imediat return!

        return best                     # nici o imbunatatire gasita
    # initPopulationMatei =====================================
