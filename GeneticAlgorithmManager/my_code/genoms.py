#!/usr/bin/python

import numpy as np
import warnings

class Genoms(object):
    """
    Clasa 'Genoms', ofera metode pentru a structura si procesa populatia.
    """
    def __init__(self, size=10, **gene_range):
        # if condition returns False, AssertionError is raised:
        #assert (isinstance(gene_range, dict)), "Parametrul 'gene_range': '{}', are un type differit de 'dict'".format(type(gene_range))

        # Define the structure: name (string), age (int), weight (float)
        self.__keys = list(gene_range.keys())
        if (len(self.__keys) == 0):
            warnings.warn("\n\nLipseste numele chromosomilor: '{}'".format(gene_range))
        # init range of genes
        self.__gene_range = gene_range
        # Define the structure: key (string), gene range (int32/float32)
        # init chromosome datatype
        self.setSize(size)

    def __getitem__(self, key):
        return self.__genoms[key]

    def __setitem__(self, key, value):
        self.__genoms[key] = value

    def __str__(self):
        info = "Genoms: shape '{}'\n".format(self.shape)
        for key in self.__keys:
            info += "\tChromosom name: '{}': range from ({} to {})".format(key, *self.__gene_range[key])
        return info

    def population(self):
        return self.__genoms

    def chromosomes(self, chromosome_name):
        return self.__genoms[chromosome_name]

    def setSize(self, size):
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
        self.__genoms = np.array(self.__new_genoms, dtype=self.__chromosome_dtype)
        # set population shape
        self.shape    = (1, len(self.__keys), (size, size))

    def setPopulationSize(self, size):
        if (self.shape[0] != size):
            del self.__new_genoms
            self.__new_genoms = []
            self.save()

    def is_genoms(self):
        return (self.__genoms.shape[0] > 0)

    def keys(self):
        return self.__keys

    def getGeneRange(self, key):
        return self.__gene_range[key]

    def concat(self, chromozomes:list):
        # salveaza chromozomii in genom
        return np.array(tuple(chromozomes), dtype=self.__chromosome_dtype)

    def append(self, genome):
        # adauga genomul in lista de genomi
        self.__new_genoms.append(genome)

    def add(self, **kw):
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
        # update shape
        self.shape = (self.__genoms.shape[0], len(self.__keys), tuple(tmp_shape))

    def help(self):
        info = """Genoms: "chromosome_name1": (min_range, max_range), "chromosome_name2": (min_range, max_range), ...\n"""
        return info
