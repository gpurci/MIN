#!/usr/bin/python

import numpy as np
from extension.mutate.my_code.mutate_base import *

class MutateScramble(MutateBase):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="MutateScramble", **configs)
        self.__fn = self._unpackMethod(method, 
                                        scramble=self.mutateScramble, 
                                        scramble_sim=self.mutateScrambleSim,
                                        scramble_diff=self.mutateScrambleDiff,
                                        mixt=self.mutateMixt)

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateScramble:
    metoda: 'scramble';      config: -> "subset_size":7;
    metoda: 'scramble_sim';  config: -> "subset_size":7;
    metoda: 'scramble_diff'; config: -> "subset_size":7;
    metoda: 'mixt';          config: -> "p_select":[1/3, 1/3, 1/3], "subset_size":7;\n"""
        print(info)

    def __mutateScramble(self, offspring, start, end):
        locuses = np.arange(start, end)
        offspring[locuses] = np.random.permutation(offspring[locuses])
        return offspring

    def mutateScramble(self, parent1, parent2, offspring, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        return self.__mutateScramble(offspring, locus, locus+subset_size)

    def mutateScrambleSim(self, parent1, parent2, offspring, subset_size=7):
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
            if (lenght > 3):
                offspring = self.__mutateScramble(offspring, start, start+lenght)
            else:
                offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        return offspring

    def mutateScrambleDiff(self, parent1, parent2, offspring, subset_size=7):
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
            if (lenght > 3):
                offspring = self.__mutateScramble(offspring, start, start+lenght)
            else:
                offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_select=None, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        elif (cond == 1):
            offspring = self.mutateScrambleSim(parent1, parent2, offspring, subset_size)
        elif (cond == 2):
            offspring = self.mutateScrambleDiff(parent1, parent2, offspring, subset_size)
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
