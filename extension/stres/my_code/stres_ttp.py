#!/usr/bin/python

import numpy as np
from extension.stres.my_code.stres_base import *

class StresTTP(StresBase):
    """
    Clasa 'StresTTP', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, subset_size=5, **configs):
        super().__init__(method, name="StresTTP", **configs)
        self.__fn = self._unpackMethod(method, 
                                        normal=self.stres, 
                                    )
        self.__score_evolution = np.zeros(subset_size, dtype=np.float32)

    def __call__(self, genoms, scores):
        return self.__fn(genoms, scores, **self._configs)

    def help(self):
        info = """StresTTP:
    metoda: 'normal'; config: -> None ;
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
