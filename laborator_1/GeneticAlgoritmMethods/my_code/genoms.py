#!/usr/bin/python

import numpy as np

class Genoms(object):
    """
    Clasa 'Genoms', ofera metode pentru a structura si procesa populatia.
    """

    def __init__(self, size=10, **gene_range):
        # if condition returns False, AssertionError is raised:
        #assert (isinstance(gene_range, dict)), "Parametrul 'gene_range': '{}', are un type differit de 'dict'".format(type(gene_range))

        # Define the structure: name (string), age (int), weight (float)
        self.__keys       = list(gene_range.keys())
        # initt range of genes
        self.__gene_range = gene_range
        # Define the structure: key (string), gene range (int32/float32)
        # init chromosome datatype
        tmp_types = []
        for key in self.__keys:
            rmin, rmax = self.__gene_range[key]
            if (isinstance(rmin, float) or isinstance(rmax, float)):
                tmp_type = (key, ("f4", size))
            else:
                tmp_type = (key, ("i4", size))
            tmp_types.append(tmp_type)
        self.__chromosome_dtype = np.dtype(tmp_types)
        # initializare genoms
        # new genoms este lista de genomuri care nu fac parte din noua generatie
        self.__new_genoms = []
        # genoms este un vector de genomuri formate, care face parte din noua generatie
        self.__genoms     = np.array(self.__new_genoms, dtype=self.__chromosome_dtype)
        self.shape        = (1, len(self.__keys), size)

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
        # update shape
        tmp_shape = []
        for key in self.__keys:
            tmp_shape.append(self.__genoms[key].shape[1])
        self.shape        = (self.__genoms.shape[0], len(self.__keys), tuple(tmp_shape))

    def help(self):
        info = """Genoms:\n"""
        return info
