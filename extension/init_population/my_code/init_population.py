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

    def __init__(self, extern_fn, genoms):
        super().__init__()
        self.__extern_fn = self.__unpack(extern_fn)
        self.__genoms  = genoms

    def __str__(self):
        info = """InitPopulation: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __call__(self, size):
        # apel direct: obiect(config)(size)
        return self.__fn(size, genoms=self.__genoms, **self.__configs)

    def __unpackMethod(self, method, extern_fn):
        # selecteaza metoda dupa care se aplica metrica
        fn = self.initPopulationAbstract
        if (method is not None):
            if   (method == "TTP_vecin"):
                fn = self.initPopulationTTP
            elif (method == "TSP_rand"):
                fn = self.initPopulationsTSPRand
            elif (method == "TTP_rand"):
                fn = self.initPopulationsTTPRand
            elif ((method == "extern") and (extern_fn is not None)):
                fn = extern_fn
        return fn

    def help(self):
        info = """InitPopulation:
    metoda: 'TTP_vecin'; config: -> "lambda_time":0.1, "vmax":1.0, "vmin":0.1, "Wmax":25936, "seed":None;
    metoda: 'TTP_rand';  config: None;
    metoda: 'TSP_rand';  config: None;
    metoda: 'extern';    config: 'extern_kw';\n"""
        return info

    def __setMethods(self, method):
        self.__method = method
        self.__extern_fn = self.__configs.pop("extern_fn", None)
        self.__fn = self.__unpackMethod(method, self.__extern_fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if (self.__extern_fn is not None):
            self.__extern_fn.setParameters(**kw)

    def initPopulationAbstract(self, size, genoms=None):
        # default: nu exista implementare
        raise NameError("Lipseste metoda '{}' pentru functia de 'InitPopulation': config '{}'".format(self.__method, self.__configs))

    # initPopulationsTSPRand -------------------------------------
    def initPopulationsTSPRand(self, population_size=-1, genoms=None):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga individ in genome
            genoms.add(tsp=np.random.permutation(individ))
        # adauga indivizi in noua generatie
        genoms.save()
        print("population {}".format(genoms.shape))
    # initPopulationsTSPRand =====================================

    # initPopulationsTTPRand -------------------------------------
    def initPopulationsTTPRand(self, population_size=-1, genoms=None):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = self.POPULATION_SIZE
        # creaza un individ
        tsp_individ = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga tsp_individ in genome
            kp_individ = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            genoms.add(tsp=np.random.permutation(tsp_individ), kp=kp_individ)
        # adauga indivizi in noua generatie
        genoms.save()
        print("population {}".format(genoms.shape))
    # initPopulationsTTPRand =====================================

    # initPopulationMatei -------------------------------------
    def initPopulationTTP(self, size, genoms=None, lambda_time=0.1,
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
            genoms.add(tsp=route, kp=kp)

            count += 1

        # finalize new generation
        genoms.save()

        print("population initialized:", genoms.shape)

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
        Greedy TTP route construction using ONE item per city:
        - tour is a permutation [c0, ..., c_{n-1}] (cycle is implied)
        - per city j we can take at most one "item" with:
            profit_j = item_profit[j]
            weight_j = item_weight[j]
        """

        start = 0
        n = self.GENOME_LENGTH

        visited = np.zeros(n, dtype=bool)
        visited[start] = True

        path = [start]
        kp = np.zeros(n, dtype=np.int32)

        cur = start
        Wcur = 0.0

        item_profit = self.item_profit
        item_weight = self.item_weight

        for _ in range(n - 1):
            cand = np.where(~visited)[0]

            # current speed
            v_cur = self.metrics.computeSpeedTTP(Wcur, vmax, vmin, Wmax)

            dist = self.distance[cur, cand]
            time_to_candidate = dist / v_cur

            # profit per candidate city = per-city profit
            profit_raw = item_profit[cand].astype(float)

            # profit after time penalty
            profit_if_take = np.maximum(
                0.0, profit_raw - lambda_time * time_to_candidate
            )

            # capacity feasibility
            can_take = (Wcur + item_weight[cand]) <= Wmax

            # heuristic score
            score = profit_if_take * can_take

            if np.all(score <= 0):
                # if all scores are zero or negative, fall back to nearest neighbor
                j = cand[np.argmin(dist)]
            else:
                order = np.argsort(score)
                top_k = min(5, len(order))
                choices = cand[order[-top_k:]]
                j = np.random.choice(choices)

            path.append(j)
            visited[j] = True

            # ---- item picking in city j ----
            tj = self.distance[cur, j] / v_cur
            p_j = item_profit[j]
            w_j = item_weight[j]

            gain = p_j - lambda_time * tj
            if gain > 0 and (Wcur + w_j) <= Wmax:
                Wcur += w_j
                kp[j] = 1

            cur = j

        # DO NOT append start again; tour is a permutation
        # path.append(path[0])  # <- removed

        return np.array(path, dtype=np.int32), kp

    # one-shot 2-opt improvement
    def _twoOpt(self, route):
        """
        single-pass 2-opt: O(N^2) candidate swaps,
        stops at the FIRST improvement found.
        """
        best = route.copy()
        best_dist = self.metrics.getIndividDistanceTTP(best, self.distance)
        n = len(best)

        # keep city 0 fixed as start
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new_route = best.copy()
                # reverse segment [i, k] inclusive
                new_route[i:k + 1] = best[i:k + 1][::-1]

                d = self.metrics.getIndividDistanceTTP(new_route, self.distance)
                if d < best_dist:
                    return new_route  # first improvement found

        return best  # no improvement
