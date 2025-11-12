#!/usr/bin/python

import numpy as np

class Genoms(object):
    """
    Clasa 'Genoms', ofera metode pentru a structura si procesa populatia.
    """

    def __init__(self, keys=[], gene_range=[], size=10):
        # if condition returns False, AssertionError is raised:
        assert (isinstance(keys, (list, tuple))), "Parametrul 'keys': '{}', are un type differit de 'list' sau 'tuple'".format(type(keys))
        assert (isinstance(gene_range, (list, tuple))), "Parametrul 'gene_range': '{}', are un type differit de 'list' sau 'tuple'".format(type(gene_range))
        assert (len(keys) == len(gene_range)), "Numarul de 'key': '{}', nu coincide cu numarul 'gene_range': {}".format(len(keys), len(gene_range))

        # Define the structure: name (string), age (int), weight (float)
        self.__keys = keys
        # initt range of genes
        tmp_dtype = np.dtype([(key, "i4") for key in keys])
        self.__gene_range       = np.array(tuple(gene_range), dtype=tmp_dtype)
        # init chromosome datatype, now is only int32
        self.__chromosome_dtype = np.dtype([(key, ("i4", size)) for key in keys])
        # initializare genoms
        # new genoms este lista de genomuri care nu fac parte din noua generatie
        self.__new_genoms = []
        # genoms este un vector de genomuri formate, care face parte din noua generatie
        self.__genoms     = np.array(self.__new_genoms, dtype=self.__chromosome_dtype)

    def __getitem__(self, key):
        return self.__genoms[key]

    def __setitem__(self, key, value):
        self.__genoms[key] = value

    def __str__(self):
        info = ""
        for genom in self.__genoms:
            info += str(genom) + "\n"
        return info

    def getKeys(self):
        return self.__keys

    def getGeneRange(self, key):
        return self.__gene_range[key]

    def append(self, **kw):
        tmp = []
        # adauga chromozomii in ordinea care au fost initializati
        for key in self.__keys:
            tmp.append(kw[key])
        # salveaza chromozomii in genom
        genome = np.array(tuple(tmp), dtype=self.__chromosome_dtype)
        # adauga genomul in lista de genomi
        self.__new_genoms.append(genome)

    def save(self):
        """Salveaza noua generatie de genomuri"""
        del self.__genoms # sterge generatia veche
        # creaza o noua generatie
        self.__genoms = np.array(self.__new_genoms, dtype=self.__chromosome_dtype)
        # initializeaza lista de genomuri
        del self.__new_genoms
        self.__new_genoms = []

    def help(self):
        info = """Genoms:\n"""
        return info
