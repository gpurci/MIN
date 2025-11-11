#!/usr/bin/python

import numpy as np
from root_GA import *

class Crossover(RootGA):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)

    def __call__(self, parent1, parent2):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self.fn(parent1, parent2)
        else: # urmasul va fi parintele 1
            offspring = parent1.copy()
        return offspring

    def __str__(self):
        info = """Crossover: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __method_fn(self):
        self.fn = self.crossoverAbstract
        if (self.__method is not None):
            if   (self.__method == "diff"):
                self.fn = self.crossoverDiff
            elif (self.__method == "split"):
                self.fn = self.crossoverSplit
            elif (self.__method == "perm_sim"):
                self.fn = self.crossoverPermSim
            elif (self.__method == "flip_sim"):
                self.fn = self.crossoverFlipSim
            elif (self.__method == "mixt"):
                self.p_mixt = [4/10, 3/10, 3/10]
                self.fn = self.crossoverMixt

        else:
            pass

    def help(self):
        info = """Crossover:
    metoda: 'diff';     config None;
    metoda: 'split';    config None;
    metoda: 'perm_sim'; config None;
    metoda: 'mixt';     config None;\n"""
        return info

    def __setMethods(self, method):
        self.__method = method
        self.__method_fn()

    def crossoverAbstract(self, parent1, parent2):
        raise NameError("Lipseste metoda '{}' pentru functia de 'Crossover': config '{}'".format(self.__method, self.__config))

    def crossoverDiff(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt diferite
        mask = parent1!=parent2
        diff_locus = np.argwhere(mask)
        size = diff_locus.shape[0]
        if (size >= 4):
            # locusurile unde genele nu coincid
            diff_locus = diff_locus.reshape(-1)
            # obtinerea genelor care nu coicid pe locusuri
            diff_genes1 = parent1[diff_locus]
            diff_genes2 = parent2[diff_locus]
            # genele care nu coincid pe pozitii
            union_genes = np.union1d(diff_genes1, diff_genes2) # valori sortate
            # adăugăm gene noi doar dacă lipsesc
            needed = diff_locus.shape[0] - union_genes.shape[0]
            if needed > 0:
                new_genes = np.random.randint(self.GENOME_LENGTH, size=needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif needed < 0:
                union_genes = union_genes[:needed]
            # permutarea genelor ce nu coincid pe pozitie
            union_genes = np.random.permutation(union_genes)
            offspring[diff_locus] = union_genes
        return offspring

    def crossoverSplit(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        start, end = np.random.randint(low=1, high=self.GENOME_LENGTH, size=2)
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
            start, lenght = Crossover.recSim(mask_sim_locus, 0, 0, 0)
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
            start, lenght = Crossover.recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def crossoverMixt(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=self.p_mixt)
        if   (cond == 0):
            offspring = self.crossoverSplit(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverPermSim(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverFlipSim(parent1, parent2)
        return offspring


    @staticmethod
    def recSim(individ, start, lenght, arg):
        if (arg < individ.shape[0]):
            tmp_arg = arg
            tmp_st  = arg
            tmp_lenght = 0
            while tmp_arg < individ.shape[0]:
                if (individ[tmp_arg] == True):
                    tmp_arg   += 1
                else:
                    tmp_lenght = tmp_arg - tmp_st
                    if (lenght < tmp_lenght):
                        start, lenght = tmp_st, tmp_lenght
                    return Crossover.recSim(individ, start, lenght, tmp_arg+1)
        else:
            return start, lenght
        return start, lenght

