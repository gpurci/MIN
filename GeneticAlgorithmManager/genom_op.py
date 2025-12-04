#!/usr/bin/python

import numpy as np
import warnings
from sys_function import sys_remove_modules

sys_remove_modules("root_GA")
from root_GA import *

def inherits_class_name(obj, class_name: str):
    return any(base.__name__ == class_name for base in obj.__class__.mro())

class GenomOp(RootGA):
    """
    Clasa 'GenomOp', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, name, **configs):
        super().__init__()
        self._genome   = genome
        self.__name    = name
        self._configs = configs
        self._method  = None
        self._man_fn  = None
        self._chromosome_fn = None
        self.__fnChromosome = None
        self.__fnGenome     = None
        self.__fnMixt       = None
        self.__fnChromosomeAbstract = None

    def __call__(self, *args):
        raise NameError("Functia '{}', lipseste implementarea: '__call__'".format(self.__name))

    def setFunctional(self, fnChromosome, fnGenome, fnMixt):
        self.__fnChromosome = fnChromosome
        self.__fnGenome     = fnGenome
        self.__fnMixt       = fnMixt

    def setAbstract(self, fnAbstract, fnChromosomeAbstract):
        self.__fnAbstract           = fnAbstract
        self.__fnChromosomeAbstract = fnChromosomeAbstract

    def __str_chromosome(self):
        info = ""
        for chrom_name in self._genome.keys():
            obj_chromosome = self._chromosome_fn[chrom_name]
            if (obj_chromosome is not None):
                tmp = "Chromozome name: '{}', info :'{}'\n".format(chrom_name, str(obj_chromosome))
            else:
                tmp = "Chromozome name: '{}', info :'{}'\n".format(chrom_name, None)
            info += "\t{}".format(tmp)
        return info

    def __str_genome(self):
        info = ""
        obj_chromosome = self._chromosome_fn["genome"]
        if (obj_chromosome is not None):
            tmp = "Genome info: '{}'\n".format(str(obj_chromosome))
        else:
            tmp = "Genome info: '{}'\n".format(None)
        info += "\t{}".format(tmp)
        return info

    def __str__(self):
        info = "{}: method '{}'\n".format(self.__name, self._method)
        if   (self._method == "chromosome"):
            info += self.__str_chromosome()
        elif (self._method == "genome"):
            info += self.__str_genome()
        elif (self._method == "mixt"):
            info += "\tp_select: {}".format(self._configs.get("p_select", None))
            info += self.__str_chromosome()
            info += self.__str_genome()
        return info

    def __unpackChromosomeMethod(self, extern_fn):
        fn = self.__fnChromosomeAbstract
        if (extern_fn is not None):
            fn = extern_fn
        if (fn is None):
            fn = self.__fnAbstract
        return fn

    def __unpackManMethod(self, method):
        fn = self.__fnAbstract
        if (method is not None):
            if   (method == "chromosome"):
                fn = self.__fnChromosome
            elif (method == "genome"):
                fn = self.__fnGenome
            elif (method == "mixt"):
                fn = self.__fnMixt
        return fn

    def help(self):
        info = """{}:
    metoda: 'chromosome'; config None;
    metoda: 'genome';     config None;
    metoda: 'mixt';       config -> "p_select":[1/2, 1/2], ;\n""".format(self.__name)
        return info

    def __unpackChromosomeConfigs(self, method):
        # unpack cromosome method
        chromosome_fn = {}
        if (method == "chromosome"):
            for key in self._genome.keys():
                extern_fn = self._configs.get(key, None)
                fn        = self.__unpackChromosomeMethod(extern_fn)
                chromosome_fn[key] = fn
        return chromosome_fn

    def __unpackGenomeConfigs(self, method):
        chromosome_fn = {}
        if (method == "genome"):
            extern_fn = self._configs.get("genome", None)
            fn        = self.__unpackChromosomeMethod(extern_fn)
            chromosome_fn["genome"] = fn
        return chromosome_fn

    def __unpackMixtConfigs(self, method):
        chromosome_fn = {}
        if (method == "mixt"):
            chromosome_fn = self.__unpackChromosomeConfigs("chromosome")
            tmp_fn        = self.__unpackChromosomeConfigs("genome")
            chromosome_fn.update(tmp_fn)
        return chromosome_fn

    def _unpackConfigs(self, method):
        # unpack manager method
        self._method = method
        self._man_fn = self.__unpackManMethod(self._method)
        # unpack cromosome method
        tmp_chr_fn = self.__unpackChromosomeConfigs(self._method)
        tmp_gen_fn = self.__unpackGenomeConfigs(self._method)
        tmp_mxt_fn = self.__unpackMixtConfigs(self._method)
        self._chromosome_fn = tmp_chr_fn
        self._chromosome_fn.update(tmp_gen_fn)
        self._chromosome_fn.update(tmp_mxt_fn)
        if (len(self._chromosome_fn.keys()) == 0):
            warnings.warn("\n\nFunctia '{}', metoda invalida '{}'".format(self.__name, self._method))

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for key in self._chromosome_fn.keys():
            obj_chromosome = self._chromosome_fn[key]
            if (obj_chromosome is not None):
                if (inherits_class_name(obj_chromosome, "RootGA")):
                    obj_chromosome.setParameters(**kw)
                else:
                    raise NameError("Functia '{}', functia externa '{}', nu mosteneste 'RootGA'".format(self.__name, obj_chromosome))
            else:
                raise NameError("Functia '{}', lipseste functia externa '{}'".format(self.__name, obj_chromosome))

    def __fnAbstract(self, **kw):
        error_mesage = "Functia '{}', lipseste metoda 'Abstract', setata in copil".format(self.__name)
        raise NameError(error_mesage)
