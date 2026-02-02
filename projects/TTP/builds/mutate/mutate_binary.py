#!/usr/bin/python

import numpy as np
from extension.mutate.mutate_base import *

class MutateBinary(MutateBase):
    """
    Clasa 'MutateBinary', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="MutateBinary", **configs)
        self.__fn = self._unpackMethod(method, 
                                        binary=self.mutateBinary, 
                                        binary_sim=self.mutateBinarySim,
                                        binary_diff=self.mutateBinaryDiff,
                                        mixt=self.mutateMixt)

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateBinary:
    metoda: 'binary';      config: None;
    metoda: 'binary_sim';  config: None;
    metoda: 'binary_diff'; config: None;
    metoda: 'mixt';        config: -> "p_select":[1/3, 1/3, 1/3];\n"""
        print(info)

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
        size_locus = mask_locus.sum()
        if (size_locus >= 4):
            start, lenght = recSim(mask_locus, 0, 0, 0)
            if (lenght > 1):
                locus = np.random.randint(low=start, high=start+lenght, size=None)
                offspring[locus] = 1 - offspring[locus]
            else:
                offspring = self.mutateBinary(parent1, parent2, offspring)
        else:
            offspring = self.mutateBinary(parent1, parent2, offspring)
        return offspring

    def mutateBinaryDiff(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_diff_locus = parent1!=parent2
        size_locus = mask_diff_locus.sum()
        if (size_locus >= 4):
            start, lenght = recSim(mask_diff_locus, 0, 0, 0)
            if (lenght > 1):
                locus = np.random.randint(low=start, high=start+lenght, size=None)
                offspring[locus] = 1 - offspring[locus]
            else:
                offspring = self.mutateBinary(parent1, parent2, offspring)
        else:
            offspring = self.mutateBinary(parent1, parent2, offspring)
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_select=None):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.mutateBinary(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateBinarySim(parent1, parent2, offspring)
        elif (cond == 2):
            offspring = self.mutateBinaryDiff(parent1, parent2, offspring)
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
