#!/usr/bin/python

import numpy as np
from extension.crossover.my_code.crossover_base import *

class CrossoverDistance(CrossoverBase):
    """
    Clasa 'CrossoverDistance', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, metrics, **configs):
        super().__init__(method, name="CrossoverDistance", **configs)
        self.__fn = self._unpackMethod(method, 
                                        single_point=self.crossoverSP, 
                                        two_point=self.crossoverTwoP,
                                        uniform=self.crossoverUniform,
                                        mixt=self.crossoverMixt)
        self.metrics = metrics

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverDistance:
    metoda: 'single_point'; config None;
    metoda: 'two_point';    config None;
    metoda: 'uniform';      config None;
    metoda: 'mixt';         config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)


    def crossoverDiff(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt diferite
        diff_locus = parent1!=parent2
        size_diff_locus = diff_locus.sum()
        if (size_diff_locus >= 4):
            # obtinerea genelor care nu coicid pe locusuri
            diff_genes1 = parent1[diff_locus]
            diff_genes2 = parent2[diff_locus]
            # genele care nu coincid pe pozitii
            union_genes = np.union1d(diff_genes1, diff_genes2) # valori sortate
            # adăugăm gene noi doar dacă lipsesc
            size_needed = size_diff_locus - union_genes.shape[0]
            if   (size_needed > 0):
                new_genes = np.random.randint(low=0, high=self.GENOME_LENGTH, size=size_needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif (size_needed < 0):
                union_genes = union_genes[:size_needed]
            # permutarea genelor ce nu coincid pe pozitie
            union_genes = np.random.permutation(union_genes)
            offspring[diff_locus] = union_genes
        return offspring

