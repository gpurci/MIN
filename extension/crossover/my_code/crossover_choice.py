#!/usr/bin/python

import numpy as np
from extension.crossover.my_code.crossover_base import *

def inherits_class_name(obj, class_name: str):
    return any(base.__name__ == class_name for base in obj.__class__.mro())

class CrossoverChoice():
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, *objects, scores=None):
        # verifica originea
        for idx, obj in enumerate(objects, 0):
            if (not inherits_class_name(obj, "CrossoverBase")):
                raise NameError("ERROR: 'CrossoverChoice' obiectul cu indexul: '{}', nu mosteneste 'CrossoverBase'".format(idx))
        # verifica scores
        if (scores is not None):
            assert (len(objects) == len(scores)), "Numarul de obiecte: '{}', este differit de numarul de score: '{}'".format(len(objects), len(scores))
            scores = np.array(scores, dtype=np.float32)
            assert ((scores.min() < 0) or (scores.max() == 0)), "Valorile scores sunt invalide, (mai mici decat '0', sau maxim '0'): '{}'".format(scores)
        else:
            scores = np.ones(len(objects), dtype=np.float32)
        # initializare
        self.__objects  = objects
        self.__p_select = scores / scores.sum()
        self.__range    = np.arange(scores.shape[0])

    def __call__(self, parent1, parent2):
        cond = np.random.choice(self.__range, size=None, p=self.__p_select)
        return self.__objects[cond](parent1, parent2)

    def __str__(self):
        info  = "CrossoverChoice: \n"
        for obj in objects:
            info += "method: '{}'\n".format(str(obj))
        return info

    def help(self):
        info = """CrossoverChoice:
    *objects - obiecte care mostenesc 'CrossoverChoice'
    scores   - valori cuprinse intre 0..10, o valoare mai mare creste probabilitatea alegerii metodei\n"""
        print(info)
