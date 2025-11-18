#!/usr/bin/python

import numpy as np

class ExternFn():
    """
    Clasa 'ExternFn', 
    """
    def __init__(self, call_fn, map_fn, dataset, **configs):
        super().__init__()
        self.__call_fn = call_fn
        self.__map_fn  = map_fn
        self.__dataset = dataset
        self.__call_configs = configs.get("call_fn", {})
        self.__list_configs = configs.get("map_fn",  {})

    def __call__(self, individ, **kw):
        kw.update(self.__call_configs)
        return self.__call_fn(individ, **kw)

    def map(self, population, **kw):
        kw.update(self.__list_configs)
        return self.__map_fn(population, **kw)

    def help(self):
        info = """ExternFn:
    'call_fn' - aplica o functie asupra unui chromosom
    'map_fn'  - aplica o functie asupra unei liste de chromosomi
    'dataset' - dataset \n"""
        return info
