#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.root_GA import *

def inherits_class_name(obj, class_name: str):
    return any(base.__name__ == class_name for base in obj.__class__.mro())

class GASelectBestBase(RootGA):
    """
    Clasa 'GASelectBestBase', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, *objects, name="GASelectBestBase", inherit_class="GABase"):
        super().__init__()
        # init
        self.__name          = name
        self.__inherit_class = inherit_class
        # verifica originea
        for idx, obj in enumerate(objects, 0):
            if (not inherits_class_name(obj, self.__inherit_class)):
                raise NameError("ERROR: '{}' obiectul cu indexul: '{}', nu mosteneste '{}'".format(self.__name, idx, self.__inherit_class))
        
        # initializare
        self._objects = objects
        self.__size   = len(self._objects)

    def __call__(self, *args):
        raise NameError("Functia '{}', lipseste implementarea: '__call__'".format(self.__name))

    def __len__(self):
        return self.__size

    def __str__(self):
        info  = "{}: \n".format(self.__name)
        for obj in objects:
            info += "method: '{}'\n".format(str(obj))
        return info

    def help(self):
        info = """{}:
    *objects - obiecte care mostenesc '{}',n""".format(self.__name, self.__inherit_class)
        print(info)
