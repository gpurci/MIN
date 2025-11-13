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
    def __init__(self, genome, **chromozoms):
        super().__init__()
        self.__genome     = genome
        self.__chromozoms = chromozoms
        self.__setMethods()

    def __call__(self, parent1, parent2):
        tmp_genome = []
        for idx, chromozome_name in enumerate(self.__genome.keys(), 0):
            tmp_genome.append(self.__call_chromozome(parent1, parent2, chromozome_name))
        return self.__genome.concat(tmp_genome)

    def __call_chromozome(self, parent1, parent2, chromozome_name):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            low, high = self.__genome.getGeneRange(chromozome_name)
            offspring = self.__fn[chromozome_name](parent1[chromozome_name], parent2[chromozome_name], 
                                        low, high, 
                                        **self.__chromozoms[chromozome_name])
        else: # urmasul va fi parintele 1
            offspring = parent1[chromozome_name].copy()
        return offspring

    def __str__(self):
        info = "Crossover:\n"
        for chrom_name in self.__genome.keys():
            tmp   = "Chromozome name: '{}', method '{}', configs: '{}'\n".format(chrom_name, self.__methods[chrom_name], self.__chromozoms[chrom_name])
            info += "\t{}".format(tmp)
        return info

    def __unpack_method(self, method):
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
    metoda: 'mixt';     config -> p_method=[4/10, 3/10, 3/10], ;\n"""
        return info

    def __setMethods(self):
        self.__fn      = {}
        self.__methods = {}
        for idx, key in enumerate(self.__genome.keys(), 0):
            method = self.__chromozoms[key].pop("method", None)
            self.__methods[key] = method
            self.__fn[key]      = self.__unpack_method(method)

    def crossoverAbstract(self, parent1, parent2, low, high):
        error_mesage = ""
        for chrom_name in self.__genome.keys():
            error_mesage += "Lipseste metoda '{}' pentru chromozomul '{}', functia de 'Crossover': config '{}'\n".format(self.__methods[chrom_name], chrom_name, self.__chromozoms[chrom_name])
        raise NameError(error_mesage)

    def crossoverDiff(self, parent1, parent2, low, high):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        low     - valoarea minima a genei
        high    - valoarea maxima a genei
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
                new_genes = np.random.randint(low=low, high=high, size=needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif needed < 0:
                union_genes = union_genes[:needed]
            # permutarea genelor ce nu coincid pe pozitie
            union_genes = np.random.permutation(union_genes)
            offspring[diff_locus] = union_genes
        return offspring

    def crossoverSplit(self, parent1, parent2, low, high):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        low     - valoarea minima a genei
        high    - valoarea maxima a genei
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

    def crossoverPermSim(self, parent1, parent2, low, high):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        low     - valoarea minima a genei
        high    - valoarea maxima a genei
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

    def crossoverFlipSim(self, parent1, parent2, low, high):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        low     - valoarea minima a genei
        high    - valoarea maxima a genei
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

    def crossoverMixt(self, parent1, parent2, low, high, p_method=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        low     - valoarea minima a genei
        high    - valoarea maxima a genei
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_method)
        if   (cond == 0):
            offspring = self.crossoverSplit(parent1, parent2, low, high)
        elif (cond == 1):
            offspring = self.crossoverPermSim(parent1, parent2, low, high)
        elif (cond == 2):
            offspring = self.crossoverFlipSim(parent1, parent2, low, high)
        return offspring


    @staticmethod
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
                    return Crossover.recSim(mask_genes, start, lenght, tmp_arg+1)
        else:
            return start, lenght
        return start, lenght

