#!/usr/bin/python

import numpy as np
from extension.stres.my_code.stres_base import *

class StresTTP(StresBase):
    """
    Clasa 'StresTTP', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset, freq_stres=10, subset_size=5, **configs):
        super().__init__(method, name="StresTTP", **configs)
        self.__fn = self._unpackMethod(method, 
                                        elite_tabu_search=self.stresTabuSearch, 
                                        elite_tabu_search_by_distance=self.stresTabuSearchDistance, 
                                    )
        self.__score_evolution = np.zeros(subset_size, dtype=np.float32)
        self.dataset    = dataset
        self.FREQ_STRES = freq_stres
        self.freq_stres = 0

    def __call__(self, genoms, scores):
        if (self.freq_stres >= self.FREQ_STRES):
            self.freq_stres = 0
            self.__fn(genoms, scores, **self._configs)
        else:
            self.freq_stres += 1

    def help(self):
        info = """StresTTP:
    metoda: 'elite_tabu_search'; config: -> None ;
    metoda: 'elite_tabu_search_by_distance'; config: -> None ;
    dataset    - setul de date de antrenare,
    freq_stres - frecventa cu care se va aplica stres,
    subset_size - esantionul de supraveghere\n"""
        print(info)

    def stresTabuSearch(self, genoms, scores):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # pack args
        args = [distance, item_profit, item_weight]

        for elite_pos in genoms.getElitePos():
            self.__tabu_search_full(genoms, elite_pos, *args)

    def stresTabuSearchDistance(self, genoms, scores):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # pack args
        args = [distance, item_profit, item_weight]

        # start tabu search by distance
        for elite_pos in genoms.getElitePos():
            self.__tabu_search_distance(genoms, elite_pos, *args)

    def __tabu_search_full(self, genoms, elite_pos, *args):
        # unpack elites
        individ = genoms[elite_pos]
        # calculeaza distanta maxima pentru normalizare
        city_d       = self.individCityDistance(individ["tsp"])
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

    def __tabu_search_distance(self, genoms, elite_pos, *args):
        # unpack elites
        individ = genoms[elite_pos]
        # calcularea distantelor dintre fiecare oras
        city_d  = self.individCityDistance(individ["tsp"])
        min_distance = city_d[city_d > 0].min()
        min_distance = min_distance if min_distance > 0 else 1
        # creare mask de depasire media pe distanta
        mask = city_d > city_d.mean()
        args_distance = np.argwhere(mask).reshape(-1)
        # compute score
        best_score   = self.__computeIndividScore(individ, min_distance, *args)
        best_individ = individ.copy()
        # apply tabu search
        for i in range(len(args_distance) - 1):
            for j in range(i + 1, len(args_distance)):
                locus1, locus2 = args_distance[i], args_distance[j]
                tmp   = individ.copy()
                tmp["tsp"][locus1], tmp["tsp"][locus2] = tmp["tsp"][locus2], tmp["tsp"][locus1]
                score = self.__computeIndividScore(tmp, min_distance, *args)
                if (score > best_score):
                    best_score   = score
                    best_individ = tmp
        # set best route
        genoms[elite_pos] = best_individ



    # ------------------ Utils ------------------
    def computeIndividDistance(self, individ):
        d = self.dataset["distance"]
        return d[individ[:-1], individ[1:]].sum() + d[individ[-1], individ[0]]

    def individCityDistance(self, individ):
        d = self.dataset["distance"]
        city_distances = d[individ[:-1], individ[1:]]
        to_first_city  = d[individ[-1], individ[0]]
        return np.concatenate((city_distances, [to_first_city]))

    def __computeScore(self, individ, *args):
        # compute distance
        profit, time, weight = self.__computeIndividAdaLiniar(individ, *args)
        time = min_norm(time)
        return (profit * time) / (profit + time)

    #  TTP Liniar ---------------------
    def __computeIndividScore(self, individ, min_distance, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01):
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
        city_d  = self.individCityDistance(tsp_individ)
        city_d  = 2*min_distance / (min_distance + city_d)
        # calculare score
        score = city_d * profits / (weights + 1e-7)
        return score.sum()


def normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_ret = (x_max-x)/(x_max-x_min+1e-7)
    return x_ret

def min_norm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 0.1
        x[:] = 0.1
    return (2*x_min)/(x+x_min)
