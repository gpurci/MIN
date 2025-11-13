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

    #initPopulationRand - ------------------------------------

    def initPopulationsTTPRand(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if population_size == -1:
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
        Generate the initial TTP population with two genomes:
            - tsp: the route (permutation of cities)
            - kp:  the picking vector (0/1)

        This function constructs the raw TSP and KP arrays.
        The GA manager is responsible for inserting them into the Genoms object.
        Returns:
            routes: np.array shape (N, GENOME_LENGTH+1)
            kps:    np.array shape (N, GENOME_LENGTH)
        """

        if seed is not None:
            np.random.seed(seed)

        # Load TTP dataset: coordinate map, distance matrix, profit, weight, etc.
        self._loadTTPdataset()

        routes = []
        kp_vectors = []
        seen = set()  # avoid duplicate TSP routes

        for _ in range(size):

            start_city = 0  # or random: np.random.randint(0, self.GENOME_LENGTH)

            # Construct greedy TTP route + KP genome
            route, kp = self._constructGreedyRoute(
                start_city, lambda_time, vmax, vmin, Wmax
            )

            # Apply 2-opt only to the route
            route = self._twoOpt(route)

            # Avoid duplicate routes
            key = tuple(route)
            if key in seen:
                continue
            seen.add(key)

            routes.append(route)
            kp_vectors.append(kp)

            if len(routes) >= size:
                break

        # Convert to numpy arrays
        routes = np.array(routes, dtype=np.int32)
        kp_vectors = np.array(kp_vectors, dtype=np.int32)

        return routes, kp_vectors

    # HELPER — incarcare dataset TTP
    def _loadTTPdataset(self):
        dataset = self.metrics.getDataset()
        self.coords      = dataset["coords"]
        self.distance    = dataset["distance"]    # matrice NxN CEIL_2D
        self.item_profit = dataset["item_profit"] # vector de profit per oras
        self.item_weight = dataset["item_weight"] # vector de weight per oras

    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        visited = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited[start] = True

        path = [start]
        kp = np.zeros(self.GENOME_LENGTH, dtype=np.int32)

        cur = start
        Wcur = 0.0
        Tcur = 0.0

        for _ in range(self.GENOME_LENGTH - 1):

            cand = np.where(~visited)[0]

            # current speed
            v_cur = self.metrics.computeSpeedTTP(Wcur, vmax, vmin, Wmax)

            # travel time to each candidate
            dist = self.distance[cur, cand]
            time_to_candidate = dist / v_cur

            # effective arrival time
            arrival_time = Tcur + time_to_candidate

            # raw profits
            profit_raw = self.item_profit[cand]

            # profit if we take the item at the arrival time
            profit_if_take = profit_raw - lambda_time * arrival_time
            profit_if_take = np.maximum(0.0, profit_if_take)

            # weight check
            can_take = (Wcur + self.item_weight[cand]) <= Wmax

            # heuristic
            score = profit_if_take * can_take

            # choose the best few randomly (diversity)
            order = np.argsort(score)
            top_k = min(5, len(order))
            choices = cand[order[-top_k:]]

            j = np.random.choice(choices)

            # update global time
            travel_time = self.distance[cur, j] / v_cur
            Tcur += travel_time

            # item decision at city j
            profit_gain = self.item_profit[j] - lambda_time * Tcur
            if profit_gain > 0.0 and (Wcur + self.item_weight[j]) <= Wmax:
                kp[j] = 1
                Wcur += self.item_weight[j]

            # accept city j
            visited[j] = True
            path.append(j)
            cur = j

        # close loop
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
