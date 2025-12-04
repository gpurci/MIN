#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class CrossoverSim(RootGA):
    """
    Clasa 'CrossoverSim', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverSim", **configs)
        self.__fn = self._unpackMethod(method, 
                                        sim=self.crossover, 
                                        scramble=self.crossoverScramble,
                                        inversion=self.crossoverInversion,
                                        mixt=self.crossoverMixt
                                    )

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverSim:
    metoda: 'sim';       config None;
    metoda: 'scramble';  config None;
    metoda: 'inversion'; config None;
    metoda: 'mixt';      config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)

    def crossover(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt similare
        mask_sim_locus = parent1==parent2
        sim_locus      = np.argwhere(mask_sim_locus)
        size           = sim_locus.shape[0]
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

    def crossoverScramble(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt similare
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.random.permutation(offspring[locuses])
        return offspring

    def crossoverInversion(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt diferite
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def crossoverMixt(self, parent1, parent2, p_select=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossover(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverScramble(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverInversion(parent1, parent2)
        return offspring

def recSim(mask_genes, start, lenght, arg):
    """Cautarea celei mai mari zone, in care genele sunt identice,
    sau cauta cea mai mare secveta de unitati"""
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

