#!/usr/bin/python

import numpy as np
import warnings

class Genoms(object):
    """
    Clasa 'Genoms', ofera metode pentru a structura si procesa populatia.
    """
    def __init__(self, genome_lenght=10, check_freq=1, elite_cmp=None, **gene_range):
        # if condition returns False, AssertionError is raised:
        #assert (isinstance(gene_range, dict)), "Parametrul 'gene_range': '{}', are un type differit de 'dict'".format(type(gene_range))

        # Define the structure: name (string), age (int), weight (float)
        self.__CHECK_FREQ = check_freq
        self.__save_count = 0
        self.__keys = list(gene_range.keys())
        if (len(self.__keys) == 0):
            warnings.warn("\n\nLipseste numele chromosomilor: '{}'".format(gene_range))
        self.__elite_cmp  = None
        self.__check_elite_cmp(elite_cmp)
        # init range of genes
        self.__gene_range = gene_range
        self.__best_chromosome = None
        self.__elites_pos = None
        self.__args_weaks_genoms = None
        self.__genoms     = None
        self.__new_genoms = None
        self.__is_cache   = False
        self.__count_cache = 0
        # Define the structure: key (string), gene range (int32/float32)
        # set population shape
        self.shape = None
        # init chromosome datatype
        self.setGenomeLenght(genome_lenght)

    def __getitem__(self, key):
        return self.__genoms[key]

    def __setitem__(self, key, value):
        self.__genoms[key] = value

    def __str__(self):
        info = "Genoms: shape: '{}', check_freq: '{}', elite_cmp: {}\n".format(self.shape, self.__CHECK_FREQ, self.__elite_cmp)
        for key in self.__keys:
            info += "\tChromosom name: '{}': range from ({} to {})\n".format(key, *self.__gene_range[key])
        return info

    def setElitePos(self, elites_pos):
        self.__elites_pos = elites_pos

    def getElitePos(self):
        return self.__elites_pos

    def getElites(self):
        return self.__genoms[self.__elites_pos]

    def getWeaksPos(self):
        return self.__args_weaks_genoms

    def setWeaksPos(self, args_weaks):
        self.__args_weaks_genoms = args_weaks

    def population(self):
        return self.__genoms

    def setPopulation(self, population=None):# TO DO: a secured set, check genom names
        if (population is not None):
            self.__update_genoms(population)
            self.__check_valid_range()

    def chromosomes(self, chromosome_name):
        return self.__genoms[chromosome_name]

    def __update_chromosome_dtype(self, size):
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

    def setGenomeLenght(self, size):
        if ((self.shape is None) or (self.shape[-1][0] != size)):
            # init chromosome datatype
            self.__update_chromosome_dtype(size)
            self.__update_genoms(np.array([], dtype=self.__chromosome_dtype))

    def setPopulationSize(self, size):
        if (self.shape[0] != size):
            pass # nu sterge populatia

    def setBest(self, chromosome):
        self.__best_chromosome = chromosome

    def getBest(self):
        return self.__best_chromosome

    def setCache(self):
        self.__is_cache = True
        self.__cache = []

    def getCache(self):
        cache = None
        if (self.__is_cache):
            self.__is_cache = False
            cache = self.__cache
            del self.__cache
            self.__cache = None
        return cache

    def isGenoms(self):
        return (self.shape[0] > 0)

    def keys(self):
        return self.__keys

    def getGeneRange(self, key):
        return self.__gene_range[key]

    def concat(self, chromosomes:list):
        # salveaza chromozomii in genom
        return np.array(tuple(chromosomes), dtype=self.__chromosome_dtype)

    def equal(self, individs, individ):
        is_equal = False
        for chromosome_name in self.__elite_cmp:
            tmps = individs[chromosome_name]
            tmp  = individ[chromosome_name]
            is_equal |= np.allclose(tmps, tmp, rtol=1e-03, atol=1e-07)
        return is_equal

    def apply(self, individs, fn, *args, **kw):
        for key in self.__keys:
            individs[key] = fn(individs[key], *args, **kw)

    def append(self, genome):
        # adauga genomul in lista de genomi
        self.__new_genoms.append(genome)

    def __unpackKW(self, **kw):
        tmp = []
        # adauga chromozomii in ordinea care au fost initializati
        for key in self.__keys:
            tmp.append(kw[key])
        # salveaza chromozomii in genom
        return np.array(tuple(tmp), dtype=self.__chromosome_dtype)

    def add(self, **kw):
        genome = self.__unpackKW(**kw)
        if (self.__is_cache):
            # adauga genomul in cache
            if (self.__cache is not None):
                self.__cache.append(genome)
            else:
                self.__cache = [genome]
        else:
            # adauga genomul in lista de genomi
            self.__new_genoms.append(genome)

    def saveInit(self):
        """Salveaza noua generatie de genomuri"""
        self.__update_genoms(np.array(self.__new_genoms, dtype=self.__chromosome_dtype))
        self.__check_valid_range()

    def save(self):
        """Salveaza noua generatie de genomuri"""
        self.__update_genoms(np.array(self.__new_genoms, dtype=self.__chromosome_dtype))
        self.__freq_check_valid_range()

    def __freq_check_valid_range(self):
        if (self.__save_count >= self.__CHECK_FREQ):
            self.__save_count = 0
            self.__check_valid_range()
        else:
            self.__save_count += 1

    def __check_valid_range(self):
        for chromosome_name in self.__keys:
            chromosomes_vals = self.__genoms[chromosome_name]
            x_min = chromosomes_vals.min()
            x_max = chromosomes_vals.max()
            r_min, r_max = self.__gene_range[chromosome_name]
            if ((x_min < r_min) or (x_max >= r_max)):
                err_msg = """Chromosomul '{}', depaseste range-ul: 
    range min: '{}', cromosome min: '{}'; 
    range max: '{}', cromosome max: '{}'""".format(chromosome_name, r_min, x_min, r_max, x_max)
                raise NameError(err_msg)

    def __update_genoms(self, population_genoms):
        del self.__genoms # sterge generatia veche
        # creaza o noua generatie
        self.__genoms = population_genoms
        # initializeaza lista de genomuri
        del self.__new_genoms
        self.__new_genoms = []
        # update shape
        self.__update_shape()

    def __update_shape(self):
        # update shape
        tmp_shape = []
        for key in self.__keys:
            tmp_shape.append(self.__genoms[key].shape[1])
        # update shape
        self.shape = (self.__genoms.shape[0], len(self.__keys), tuple(tmp_shape))

    def __check_elite_cmp(self, elite_cmp):
        if (elite_cmp is None):
            self.__elite_cmp = self.__keys
        else:
            if (isinstance(elite_cmp, list)):
                for key in elite_cmp:
                    err_msg = "Numele chromosomului: '{}', pentru selectia 'elitei', nu este 'string'".format(key)
                    assert (isinstance(key, str)), err_msg
                    err_msg = "Numele chromosomului: '{}', pentru selectia 'elitei', nu este in lista de nume '{}' a chromosomilor".format(key, self.__keys)
                    assert (key in self.__keys), err_msg
                self.__elite_cmp = elite_cmp
            elif (isinstance(elite_cmp, str)):
                err_msg = "Numele chromosomului: '{}', pentru selectia 'elitei', nu este in lista de nume '{}' a chromosomilor".format(elite_cmp, self.__keys)
                assert (elite_cmp in self.__keys), err_msg
                self.__elite_cmp = [elite_cmp]
            else:
                err_msg = "Numele chromosomului: '{}', pentru selectia 'elitei', este differit de type 'list' sau 'str': type {}".format(elite_cmp, type(elite_cmp))
                raise NameError(err_msg)

    def help(self):
        info = """Genoms: 
    genome_lenght - lungimea genomului, 
    check_freq    - frecventa (epoch/generatii) cu care chromosomii vor fi verificati ca sunt in range-ul prestabilit, 
    elite_cmp     - chromosomii dupa care se compara doi genomi pentru a stabili, data doi genomi sunt egali, 
    "chromosome_name1": (min_range, max_range), "chromosome_name2": (min_range, max_range), ...\n"""
        return info