#!/usr/bin/python

import numpy as np
from extension.stres.stres_base import *
from extension.utils.insertion import *

class StresTTPV1(StresBase):
    """
    Clasa 'StresTTPV1', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man, freq_stres=10, **configs):
        super().__init__(method, name="StresTTPV1", **configs)
        self.__fn = self._unpackMethod(method, 
                                        insertion_tabu=self.stresInsertionTabuSearch, 
                                        rand_tabu=self.stresRandTabuSearch, 
                                    )
        self.dataset_man = dataset_man
        self.FREQ_STRES  = freq_stres
        self.freq_stres  = 0

    def __call__(self, genoms, scores):
        if (self.freq_stres < self.FREQ_STRES):
            self.freq_stres += 1
        else:
            self.freq_stres = 1
            self.__fn(genoms, scores, **self._configs)

    def help(self):
        info = """StresTTPV1:
    metoda: 'insertion_tabu'; config: -> v_min=0.1, v_max=1, W=2000, R=1 ;
    dataset_man - manager la setul de date de antrenare,
    freq_stres  - frecventa cu care se va aplica stres,\n"""
        print(info)

    def stresInsertionTabuSearch(self, genoms, scores, v_min=0.1, v_max=1, W=2000, R=1):
        # start tabu search by distance
        for elite_pos in genoms.getElitePos():
            individ = genoms[elite_pos]
            tsp_individ, kp_individ = individ["tsp"], individ["kp"]
            tsp_individ = self.insertion_tabu_search_distance(tsp_individ)
            kp_individ  = self.tabu_search_score(tsp_individ, kp_individ, v_min, v_max, W, R)
            genoms[elite_pos]["tsp"] = tsp_individ
            genoms[elite_pos]["kp"]  = kp_individ

    def stresRandTabuSearch(self, genoms, scores, v_min=0.1, v_max=1, W=2000, R=1):
        # start tabu search by distance
        for elite_pos in genoms.getElitePos():
            individ = genoms[elite_pos]
            tsp_individ, kp_individ = individ["tsp"], individ["kp"]
            tsp_individ = self.insertion_tabu_search_distance(tsp_individ)
            kp_individ  = self.rand_tabu_search_score(tsp_individ, kp_individ, v_min, v_max, W, R)
            genoms[elite_pos]["tsp"] = tsp_individ
            genoms[elite_pos]["kp"]  = kp_individ

    def insertion_tabu_search_distance(self, tsp_individ):
        # calcularea distantelor dintre fiecare oras
        city_distances = self.dataset_man.individCityDistance(tsp_individ)
        # creare mask de depasire media pe distanta
        locus1 = (np.argmax(city_distances) + 1) % self.GENOME_LENGTH
        # compute score
        best_distance = city_distances.sum()
        best_individ  = tsp_individ.copy()
        # apply tabu search
        locusses = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        locusses = np.delete(locusses, obj=[locus1])
        for locus2 in locusses:
            tmp = insertion(tsp_individ.copy(), locus1, locus2)
            distance = self.dataset_man.computeIndividDistance(tmp)
            if (distance < best_distance):
                best_distance = distance
                best_individ  = tmp
        # set best route
        return best_individ

    def tabu_search_score(self, tsp_individ, kp_individ, v_min, v_max, W, R):
        # calcularea distantelor dintre fiecare oras
        mask = kp_individ==1
        arg_take   = np.argwhere(mask).reshape(-1)
        arg_notake = np.argwhere(np.invert(mask)).reshape(-1)
        # compute score
        best_score   = self.dataset_man.computeIndividScore(tsp_individ, kp_individ, v_min=v_min, v_max=v_max, W=W, R=R)
        best_individ = kp_individ.copy()
        # apply tabu search
        for locus1 in arg_take:
            for locus2 in arg_notake:
                tmp = kp_individ.copy()
                tmp[locus1] = 0
                tmp[locus2] = 1
                score = self.dataset_man.computeIndividScore(tsp_individ, tmp, v_min=v_min, v_max=v_max, W=W, R=R)
                if (score > best_score):
                    best_score   = score
                    best_individ = tmp.copy()
        # set best route
        return best_individ

    def rand_tabu_search_score(self, tsp_individ, kp_individ, v_min, v_max, W, R):
        # calcularea distantelor dintre fiecare oras
        mask = kp_individ==1
        arg_take   = np.argwhere(mask).reshape(-1)
        arg_notake = np.argwhere(np.invert(mask)).reshape(-1)
        # compute score
        best_score   = self.dataset_man.computeIndividScore(tsp_individ, kp_individ, v_min=v_min, v_max=v_max, W=W, R=R)
        best_individ = kp_individ.copy()
        # apply tabu search
        p_select = np.arange(1, arg_take.shape[0]+1, dtype=np.float32)
        p_select = p_select / p_select.sum()
        locus1 = np.random.choice(arg_take, size=None, p=p_select)
        for locus2 in arg_notake:
            tmp = kp_individ.copy()
            tmp[locus1] = 0
            tmp[locus2] = 1
            score = self.dataset_man.computeIndividScore(tsp_individ, tmp, v_min=v_min, v_max=v_max, W=W, R=R)
            if (score > best_score):
                best_score   = score
                best_individ = tmp.copy()
        # set best route
        return best_individ

    # ------------------ Utils ------------------