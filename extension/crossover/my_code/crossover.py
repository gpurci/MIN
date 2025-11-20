#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class Crossover(RootGA):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__()
        self.__method  = method
        self.__configs = configs
        self.__fn = self.__unpackMethod(method)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self.__configs)

    def __str__(self):
        info  = "Crossover: method '{}'\n".format(self.__method)
        tmp   = "configs: '{}'\n".format(self.__configs)
        info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method):
        fn = self.crossoverAbstract
        if (method is not None):
            if   (method == "diff"):
                fn = self.crossoverDiff
            elif (method == "split"):
                fn = self.crossoverSplit
            elif (method == "perm_sim"):
                fn = self.crossoverPermSim
            elif (method == "flip_sim"):
                fn = self.crossoverFlipSim
            elif (method == "mixt"):
                fn = self.crossoverMixt

        return fn

    def help(self):
        info = """Crossover:
    metoda: 'diff';     config None;
    metoda: 'split';    config None;
    metoda: 'perm_sim'; config None;
    metoda: 'flip_sim'; config None;
    metoda: 'mixt';     config -> "p_method":[1/4, 1/4, 1/4, 1/4], ;\n"""
        return info

    def crossoverAbstract(self, parent1, parent2):
        error_mesage = "Functia 'Crossover', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

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

    def crossoverSplit(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        start, end = np.random.randint(low=0, high=self.GENOME_LENGTH, size=2)
        # corectie diapazon
        if (start > end): start, end = end, start
        # copierea rutei din cel de al doilea parinte
        offspring[start:end] = parent2[start:end]
        return offspring

    def crossoverPermSim(self, parent1, parent2):
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
                offspring[locuses] = np.random.permutation(offspring[locuses])
        return offspring

    def crossoverFlipSim(self, parent1, parent2):
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

    def crossoverMixt(self, parent1, parent2, p_method=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2, 3], size=None, p=p_method)
        if   (cond == 0):
            offspring = self.crossoverSplit(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverDiff(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverPermSim(parent1, parent2)
        elif (cond == 3):
            offspring = self.crossoverFlipSim(parent1, parent2)
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

