#!/usr/bin/python

import numpy as np
from extension.stres.stres_base import *
from extension.utils.normalization import *

class StresTTPV1(StresBase):
    """
    Clasa 'StresTTPV1', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man, freq_stres=10, **configs):
        super().__init__(method, name="StresTTPV1", **configs)
        self.__fn = self._unpackMethod(method, 
                                        elite_tabu_search=self.stresTabuSearch, 
                                        elite_tabu_search_by_distance=self.stresTabuSearchDistance, 
                                    )
        self.dataset_man = dataset_man
        self.FREQ_STRES  = freq_stres
        self.freq_stres  = 0

    def __call__(self, genoms, scores):
        if (self.freq_stres < self.FREQ_STRES):
            self.freq_stres += 1
        else:
            self.freq_stres = 0
            self.__fn(genoms, scores, **self._configs)

    def help(self):
        info = """StresTTPV1:
    metoda: 'normal';                        config: -> None ;
    metoda: 'elite_tabu_search';             config: -> None ;
    metoda: 'elite_tabu_search_by_distance'; config: -> None ;
    dataset_man - manager la setul de date de antrenare,
    freq_stres  - frecventa cu care se va aplica stres,
    subset_size - esantionul de supraveghere\n"""
        print(info)

    def stresTabuSearch(self, genoms, scores):
        # unpack datassets
        distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
        # pack args
        args = [distance, item_profit, item_weight]

        for elite_pos in genoms.getElitePos():
            self.__tabu_search_full(genoms, elite_pos, *args)

    def stresTabuSearchDistance(self, genoms, scores, subset_size=10):
        # unpack datassets
        distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
        # pack args
        args = [distance, item_profit, item_weight]

        # start tabu search by distance
        for elite_pos in genoms.getElitePos():
            self.__tabu_search_distance(genoms, elite_pos, subset_size, *args)

    def __tabu_search_full(self, genoms, elite_pos, *args):
        # unpack elites
        individ = genoms[elite_pos]
        # calculeaza distanta maxima pentru normalizare
        city_d       = self.dataset_man.individCityDistance(individ["tsp"])
        min_distance = city_d[city_d > 0].min()
        min_distance = min_distance if min_distance > 0 else 1
        # compute score
        best_score   = self.__computeIndividScore(individ, min_distance, *args)
        best_individ = individ.copy()
        # apply tabu search
        for locus1 in range(self.GENOME_LENGTH-1):
            for locus2 in range(locus1+1, self.GENOME_LENGTH):
                tmp   = individ.copy()
                tmp["tsp"][locus1], tmp["tsp"][locus2] = tmp["tsp"][locus2], tmp["tsp"][locus1]
                score = self.__computeIndividScore(tmp, min_distance, *args)
                if (score > best_score):
                    best_score   = score
                    best_individ = tmp
        # set best route
        genoms[elite_pos] = best_individ

    def __tabu_search_distance(self, genoms, elite_pos, subset_size, *args):
        # unpack elites
        individ = genoms[elite_pos]
        # calcularea distantelor dintre fiecare oras
        city_d  = self.dataset_man.individCityDistance(individ["tsp"])
        min_distance = city_d[city_d > 0].min()
        min_distance = min_distance if min_distance > 0 else 1
        # creare mask de depasire media pe distanta
        argmax_distance = np.argmax(city_d)
        # seteaza un camp de actiune
        start = max(0,                  argmax_distance-subset_size)
        stop  = min(self.GENOME_LENGTH, argmax_distance+subset_size)
        # compute score
        best_score   = self.__computeIndividScore(individ, min_distance, *args)
        best_individ = individ.copy()
        # apply tabu search
        for locus1 in range(start,        stop-1, 1): # exclude ultimul
            for locus2 in range(locus1+1, stop,   1):
                tmp   = individ.copy()
                tmp["tsp"][locus1], tmp["tsp"][locus2] = tmp["tsp"][locus2], tmp["tsp"][locus1]
                score = self.__computeIndividScore(tmp, min_distance, *args)
                if (score > best_score):
                    best_score   = score
                    best_individ = tmp
        # set best route
        genoms[elite_pos] = best_individ

    # ------------------ Utils ------------------
    def __computeIndividScore(self, individ, min_distance, *args):
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack args
        # unpack datassets
        distance, item_profit, item_weight = args
        # obtine 
        takes   = kp_individ[tsp_individ]
        # calculare profit si greutate pentru fiecare oras
        profits = item_profit[tsp_individ]*takes
        weights = item_weight[tsp_individ]*takes
        # calcularea distantelor dintre fiecare oras
        city_d  = self.dataset_man.individCityDistance(tsp_individ)
        city_d  = 2*min_distance / (min_distance + city_d)
        # calculare score
        score = city_d * profits / (weights + 1e-7)
        return score.sum()
