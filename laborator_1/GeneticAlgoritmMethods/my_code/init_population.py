#!/usr/bin/python

import numpy as np
from my_code.root_GA import *
from  my_code.metrics import Metrics

class InitPopulation(RootGA):
    """
    Generează populația inițială TTP.
    Pentru fiecare individ se alege un oraș de start random și se construiește ruta
    alegând la fiecare pas următorul oraș în funcție de un scor simplu:
        scor = profit - λ * timp_de_deplasare
    După construirea rutei se aplică o singură îmbunătățire 2-opt (+ eliminare duplicate).
    Returnează un array de rute valide (start == end).
    """

    def __init__(self, config, metrics):
        # metrics = obiectul Metrics — folosit pentru dataset
        self.metrics = metrics
        self.setConfig(config)

    def __call__(self, size):
        # apel direct: obiect(config)(size)
        return self.fn(size)

    def __config_fn(self):
        # selecteaza implementarea reala in functie de config
        self.fn = self.initPopulationAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                # folosim versiunea ta (Matei)
                self.fn = self.initPopulationMatei
        else:
            pass

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def initPopulationAbstract(self, size):
        # default: nu exista implementare
        raise NameError("Lipseste configuratia pentru functia de 'InitPopulation': config '{}'".format(self.__config))

    def initPopulationMatei(self,
                            size=2000, lambda_time=0.1,
                            vmax=1.0, vmin=0.1, Wmax=25936, seed=None):
        """
        Genereaza `size` indivizi folosind o euristica greedy TTP:
        - fiecare individ incepe dintr-un oras random
        - la fiecare pas alegem urmatorul oras dupa: profit - λ * timp_de_calatorie
        - dupa ce rute se construiesc → aplicam 2-opt simplu
        """

        if seed is not None:
            np.random.seed(seed)

        # incarcam coordonate / distante / profit / weight
        self._loadTTPdataset()

        population, seen = [], set()

        # alege start random pentru fiecare individ
        starts = np.random.randint(0, self.GENOME_LENGTH, size=size)

        for s in starts:
            # construieste 1 ruta
            path_np = self._constructGreedyRoute(s, lambda_time, vmax, vmin, Wmax)
            # aplica o singura iteratie 2-opt (accelerat)
            path_np = self.__twoOpt(path_np)

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
        cur  = start
        Wcur = 0.0  # current knapsack weight

        # GENOME_LENGTH - 1 mutări (ultima este întoarcerea spre start)
        for _ in range(self.GENOME_LENGTH-1):
            # lista orașelor nevizitate
            cand = np.where(~visited)[0]

            # viteza actuala in functie de cat ai incarcat rucsacul
            v_cur = self._computeSpeed(Wcur, vmax, vmin, Wmax)

            # calculeaza durata de calatorie la fiecare oras candidat
            dist  = self.distance[cur, cand]
            time  = dist / v_cur

            # vezi daca putem lua item-ul (nu depasim capacitatea)
            can_take = (Wcur + self.item_weight[cand]) <= Wmax
            profit   = self.item_profit[cand] * can_take

            # scor euristic: profit - lambda * timp
            score = profit - lambda_time * time

            # alegem din top 5 scoruri cele mai bune → alegere random din top
            order   = np.argsort(score)
            top_k   = min(5, len(order))
            choices = cand[order[-top_k:]]

            # alegem un oras random din cele mai bune
            j = np.random.choice(choices)
            path.append(j)
            visited[j] = True

            # daca incape — actualizam greutatea
            if (Wcur + self.item_weight[j]) <= Wmax:
                Wcur += self.item_weight[j]

            cur = j

        # inchide ciclul
        path.append(path[0])
        return np.array(path, dtype=np.int32)

    # one-shot 2-opt improvement
    def __twoOpt(self, route):
        """
        single-pass 2-opt: testeaza O(N^2) swap-uri
        si se opreste la PRIMA imbunatatire gasita.
        """
        best = route.copy()
        best_dist = self.___getIndividDistanceTTP(best)
        n = len(route) - 1

        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new_route = best.copy()
                new_route[i:k] = best[k-1:i-1:-1]

                d = self.___getIndividDistanceTTP(new_route)
                if d < best_dist:
                    return new_route     # improvement found — imediat return!

        return best                     # nici o imbunatatire gasita
    
    # calculeaza viteza TTP in functie de weight
    def _computeSpeed(self, Wcur, vmax, vmin, Wmax):
        frac = min(1.0, Wcur/Wmax)
        v = vmax - frac*(vmax-vmin)   # formula TTP standard
        if v < 1e-9:
            v = 1e-9
        return v
    
    # calculeaza lungimea rutei (in TTP)
    def ___getIndividDistanceTTP(self, individ):
        """Calculul distantei rutelor"""
        distances = self.distance[individ[:-1], individ[1:]]
        distance = distances.sum() + self.distance[individ[-1], individ[0]]
        return distance