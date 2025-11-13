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
        # creaza o populatie aleatorie
        for _ in range(population_size):
            # adauga tsp_individ in genome
            kp_individ = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
            self.__genoms.add(tsp=np.random.permutation(tsp_individ), kp=kp_individ)
        # adauga indivizi in noua generatie
        self.__genoms.save()
        print("population {}".format(self.__genoms.shape))
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

        # incarcam coordonate / distante / profit / weight
        self._loadTTPdataset()

        population, seen = [], set()

        # start din primul oras
        starts = np.zeros(size, dtype=int)

        for s in starts:
            # construieste 1 ruta
            path_np = self._constructGreedyRoute(s, lambda_time, vmax, vmin, Wmax)
            # aplica o singura iteratie 2-opt (accelerat)
            path_np = self._twoOpt(path_np)

            tup = tuple(path_np)
            if tup in seen:
                # evita rute duplicat
                continue

            seen.add(tup)
            population.append(path_np)

            if len(population) >= size:
                break

        return np.array(population, dtype=np.int32)

    # HELPER — incarcare dataset TTP
    def _loadTTPdataset(self):
        dataset = self.metrics.getDataset()
        self.coords      = dataset["coords"]
        self.distance    = dataset["distance"]    # matrice NxN CEIL_2D
        self.item_profit = dataset["item_profit"] # vector de profit per oras
        self.item_weight = dataset["item_weight"] # vector de weight per oras

    # construieste o ruta greedy pornind dintr-un oras
    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        visited = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited[start] = True

        path = [start]
        cur = start
        Wcur = 0.0  # current knapsack weight

        # GENOME_LENGTH - 1 mutări (ultima este întoarcerea spre start)
        for _ in range(self.GENOME_LENGTH - 1):

            cand = np.where(~visited)[0]

            # viteza actuala in functie de cat ai incarcat rucsacul
            v_cur = self.metrics.computeSpeedTTP(Wcur, vmax, vmin, Wmax)
            dist = self.distance[cur, cand]
            time = dist / v_cur

            # profit brut al itemului din orașul candidat
            profit_raw = self.item_profit[cand]

            # calculăm cât profit real rămâne după penalizarea de timp
            # dacă e negativ -> îl forțăm la 0 (adică nu merită să îl luăm)
            profit_if_take = profit_raw - lambda_time * time
            profit_if_take = np.maximum(0.0, profit_if_take)

            # putem lua item-ul DOAR dacă încăperea / capacitatea nu este depășită
            can_take = (Wcur + self.item_weight[cand]) <= Wmax

            # scor euristic: profit efectiv după penalizare * dacă avem voie să îl luăm
            score = profit_if_take * can_take  # <--- asta decide urmatorul oraș

            # alegem din top 5 scoruri cele mai bune → alegere random din top
            order = np.argsort(score)
            top_k = min(5, len(order))
            choices = cand[order[-top_k:]]

            j = np.random.choice(choices)
            path.append(j)
            visited[j] = True

            # la commit-ul final decidem efectiv dacă luăm item-ul:
            # dacă profitul de după penalizare e pozitiv și încape în rucsac
            pj_raw = self.item_profit[j]
            dist_j = self.distance[cur, j]
            time_j = dist_j / v_cur
            profit_gain = pj_raw - lambda_time * time_j

            if profit_gain > 0.0 and (Wcur + self.item_weight[j]) <= Wmax:
                Wcur += self.item_weight[j]

            cur = j

        # inchide ciclul
        path.append(path[0])
        return np.array(path, dtype=np.int32)

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
