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
                                        normal=self.stres, 
                                        elite_tabu_search=self.stresTabuSearch, 
                                        elite_tabu_search_by_distance=self.stresTabuSearchDistance, 
                                    )
        self.__score_evolution = np.zeros(subset_size, dtype=np.float32)
        self.dataset    = dataset
        self.FREQ_STRES = freq_stres
        self.freq_stres = 0

    def __call__(self, genoms, scores):
        return self.__fn(genoms, scores, **self._configs)

    def help(self):
        info = """StresTTP:
    metoda: 'normal';            config: -> None ;
    metoda: 'elite_tabu_search'; config: -> None ;
    metoda: 'elite_tabu_search_by_distance'; config: -> None ;
    dataset    - setul de date de antrenare,
    freq_stres - frecventa cu care se va aplica stres,
    subset_size - esantionul de supraveghere\n"""
        print(info)

    def stres(self, genoms, scores):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        scores - scorul evolutiei
        """
        check_distance = np.allclose(self.__score_evolution.mean(), scores["score"], rtol=1e-01, atol=1e-03)
        #print("distance evolution {}, distance {}".format(check_distance, best_distance))

        # genoms.getElitePos() = returneaza pozitiile elite

        if (check_distance):
            self.__score_evolution[:] = 0
            print("scores {}".format(scores))
            # ADD STRESS FUNCTION
        else:
            self.__score_evolution[:-1] = self.__score_evolution[1:]
            self.__score_evolution[-1]  = scores["score"]

    def stresTabuSearch(self, genoms, scores):
        if (self.freq_stres >= self.FREQ_STRES):
            for elite_pos in genoms.getElitePos():
                # unpack elites
                individ = genoms[elite_pos]
                # unpack route
                route   = individ["tsp"]
                #print("route", route)
                # compute distance
                best_distance = self.computeIndividDistance(route)
                best_route    = route.copy()
                # apply tabu search
                for locus1 in range(self.GENOME_LENGTH-1):
                    for locus2 in range(locus1+1, self.GENOME_LENGTH):
                        tmp = route.copy()
                        tmp[locus1], tmp[locus2] = tmp[locus2], tmp[locus1]
                        distance = self.computeIndividDistance(tmp)
                        if (distance < best_distance):
                            best_distance = distance
                            best_route    = tmp
                # set best route
                genoms[elite_pos]["tsp"] = best_route
                #print("genoms", genoms[elite_pos]["tsp"])
        else:
            self.freq_stres += 1

    def stresTabuSearchDistance(self, genoms, scores):
        if (self.freq_stres >= self.FREQ_STRES):
            # start tabu search by distance
            for elite_pos in genoms.getElitePos():
                # unpack elites
                individ = genoms[elite_pos]
                # unpack route
                route   = individ["tsp"]
                # calcularea distantelor dintre fiecare oras
                city_d  = self.individCityDistance(route)
                # creare mask de depasire media pe distanta
                mask    = city_d > city_d.mean()
                args    = np.argwhere(mask).reshape(-1)

                # compute distance
                best_distance = city_d.sum()
                best_route    = route.copy()
                # apply tabu search
                for i in range(len(args) - 1):
                    for j in range(i + 1, len(args)):
                        locus1, locus2 = args[i], args[j]
                        tmp = route.copy()
                        tmp[locus1], tmp[locus2] = tmp[locus2], tmp[locus1]
                        distance = self.computeIndividDistance(tmp)
                        if (distance < best_distance):
                            best_distance = distance
                            best_route    = tmp
                # set best route
                genoms[elite_pos]["tsp"] = best_route
        else:
            self.freq_stres += 1


    # ------------------ Utils ------------------
    def computeIndividDistance(self, individ):
        d = self.dataset["distance"]
        return d[individ[:-1], individ[1:]].sum() + d[individ[-1], individ[0]]

    def individCityDistance(self, individ):
        d = self.dataset["distance"]
        city_distances = d[individ[:-1], individ[1:]]
        to_first_city  = d[individ[-1], individ[0]]
        return np.concatenate((city_distances, [to_first_city]))
        """

            def _cost(self, route):
                d = self.dist
                return d[route[:-1], route[1:]].sum() + d[route[-1], route[0]]
        """
