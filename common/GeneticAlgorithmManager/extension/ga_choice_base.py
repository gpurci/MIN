#!/usr/bin/python

import numpy as np
from extension.ga_base import *

def inherits_class_name(obj, class_name: str):
    return any(base.__name__ == class_name for base in obj.__class__.mro())

class GAChoiceBase():
    """
    """
    def __init__(self, *objects, p_select=None, scores=None, name="GAChoiceBase", inherit_class="GABase"):
        # init
        self.__name          = name
        self.__inherit_class = inherit_class
        # verifica originea
        for idx, obj in enumerate(objects, 0):
            if (not inherits_class_name(obj, self.__inherit_class)):
                raise NameError("ERROR: '{}' obiectul cu indexul: '{}', nu mosteneste '{}'".format(self.__name, idx, self.__inherit_class))
        
        if (p_select is not None):
            error_msg = "Numarul de obiecte: '{}', este differit de numarul de 'p_select': '{}'".format(len(objects), len(p_select))
            assert (len(objects) == len(p_select)), error_msg
            p_select = np.array(p_select, dtype=np.float32)
            error_msg = "Valorile 'p_select' sunt invalide, (mai mici decat '0', maxim '0' sau max > '1'): '{}'".format(p_select)
            assert ((p_select.min() < 0) or (p_select.max() == 0) or (p_select.max() > 1)), error_msg
        else:
            # verifica scores
            if (scores is not None):
                error_msg = "Numarul de obiecte: '{}', este differit de numarul de 'scores': '{}'".format(len(objects), len(scores))
                assert (len(objects) == len(scores)), error_msg
                scores = np.array(scores, dtype=np.float32)
                error_msg = "Valorile 'scores' sunt invalide, (mai mici decat '0', sau maxim '0'): '{}'".format(scores)
                assert ((scores.min() < 0) or (scores.max() == 0)), error_msg
                # calculeaza probabilitatea dupa valorile scores
                p_select = scores / scores.sum()
            else:
                # calculeaza probabilitatea cu sanse egale de selectie
                p_select = np.ones(len(objects), dtype=np.float32) / len(objects)
        # initializare
        self._objects  = objects
        self._p_select = p_select
        self._range    = np.arange(p_select.shape[0])

    def __call__(self, *args):
        raise NameError("Functia '{}', lipseste implementarea: '__call__'".format(self.__name))

    def __str__(self):
        info  = "{}: \n".format(self.__name)
        for obj in objects:
            info += "method: '{}'\n".format(str(obj))
        return info

    def help(self):
        info = """{}:
    *objects - obiecte care mostenesc '{}',
    p_select - probabilitatea de selectie a unei metode, 0...1
    scores   - probabilitatea de selectie a unei metode, valori cuprinse intre 0...10, se normalizeaza!\n""".format(self.__name, self.__inherit_class)
        print(info)
