#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class Mutate(RootGA):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__()
        self.__method  = method
        self.__configs = configs
        self.__fn = self.__unpackMethod(method)

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self.__configs)

    def __str__(self):
        info  = "Mutate: method '{}'\n".format(self.__method)
        tmp   = "configs: '{}'\n".format(self.__configs)
        info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method):
        fn = self.mutateAbstract
        if (method is not None):
            if   (method == "binary"):
                fn = self.mutateBinary
            elif (method == "binary_sim"):
                fn = self.mutateBinarySim
            elif (method == "binary_diff"):
                fn = self.mutateBinaryDiff
            elif (method == "mixt_binary"):
                fn = self.mutateMixtBinary

        return fn

    def help(self):
        info = """Mutate:
    metoda: 'binary';     config: None;
    metoda: 'binary_sim'; config: None;
    metoda: 'binary_diff';config: None;
    metoda: 'mixt_binary';config: -> "p_method":[1/3, 1/3, 1/3], "subset_size":7;\n"""
        return info

    def mutateAbstract(self, parent1, parent2, offspring):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

    def mutateBinary(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
        offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateBinarySim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_locus = parent1==parent2
        sim_locus = np.argwhere(mask_locus)
        size_locus = mask_locus.sum()
        if (size_locus >= 4):
            start, lenght = recSim(mask_locus, 0, 0, 0)
            if (lenght > 1):
                locus = np.random.randint(low=start, high=start+lenght, size=None)
                offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateBinaryDiff(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_diff_locus = parent1!=parent2
        sim_locus = np.argwhere(mask_diff_locus)
        size_locus = mask_diff_locus.sum()
        if (size_locus >= 4):
            start, lenght = recSim(mask_diff_locus, 0, 0, 0)
            if (lenght > 1):
                locus = np.random.randint(low=start, high=start+lenght, size=None)
                offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateMixtBinary(self, parent1, parent2, offspring, p_method=None):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2, 3, 4], size=None, p=p_method)
        return offspring

def recSim(mask_genes, start, lenght, arg):
    """Cautarea celei mai mari zone, in care genele sunt identice"""
    if (arg < mask_genes.shape[0]):
        tmp_arg = arg
        tmp_st  = arg
        tmp_lenght = 0
        while tmp_arg < mask_genes.shape[0]:
            if (mask_genes[tmp_arg]):
                tmp_arg   += 1
            else:
                tmp_lenght = tmp_arg - tmp_st
                if (lenght < tmp_lenght):
                    start, lenght = tmp_st, tmp_lenght
                return recSim(mask_genes, start, lenght, tmp_arg+1)
    else:
        return start, lenght
    return start, lenght
