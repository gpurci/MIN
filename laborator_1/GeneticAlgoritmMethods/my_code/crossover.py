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
    def __init__(self, genome, **chromosom_configs):
        super().__init__()
        self.__genome = genome
        self.__chromosom_configs = chromosom_configs
        self.__unpackConfigs()

    def __call__(self, parent1, parent2):
        tmp_genome = []
        self.__select_parent_chromosome = 0
        for idx, chromozome_name in enumerate(self.__genome.keys(), 0):
            chromosome_val = self.__crossover_chromosome(parent1, parent2, chromozome_name)
            tmp_genome.append(chromosome_val)
        return self.__genome.concat(tmp_genome)

    def __crossover_chromosome(self, parent1, parent2, chromozome_name):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            self.__extern_fn = self.__externs_fn[chromozome_name]
            low, high = self.__genome.getGeneRange(chromozome_name)
            offspring = self.__fn[chromozome_name](parent1[chromozome_name], parent2[chromozome_name], 
                                        low, high, 
                                        **self.__chromosom_configs[chromozome_name])
        else: # selectie chromosom intreg
            if (self.__select_parent_chromosome == 0): # mosteneste chromosome parinte 1
                offspring = parent1[chromozome_name].copy()
                self.__select_parent_chromosome = 1
            else: # mosteneste chromosome parinte 2
                offspring = parent2[chromozome_name].copy()
                self.__select_parent_chromosome = 0
        return offspring

    def __str__(self):
        info = "Crossover:\n"
        for chrom_name in self.__genome.keys():
            tmp   = "Chromozome name: '{}', method '{}', configs: '{}'\n".format(chrom_name, self.__methods[chrom_name], self.__chromosom_configs[chrom_name])
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
    metoda: 'flip_sim'; config None;
    metoda: 'mixt';     config -> "p_method":[1/4, 1/4, 1/4, 1/4], ;\n"""
        return info

    def __unpackConfigs(self):
        self.__fn      = {}
        self.__methods = {}
        self.__externs_fn = {}
        for idx, key in enumerate(self.__genome.keys(), 0):
            method = self.__chromosom_configs[key].pop("method", None)
            self.__methods[key] = method
            extern_fn = self.__chromosom_configs[key].pop("extern_fn", None)
            self.__externs_fn[key] = extern_fn
            self.__fn[key]      = self.__unpack_method(method)

    def crossoverAbstract(self, parent1, parent2, low, high):
        error_mesage = ""
        for chrom_name in self.__genome.keys():
            error_mesage += "Lipseste metoda '{}' pentru chromozomul '{}', functia de 'Crossover': config '{}'\n".format(self.__methods[chrom_name], chrom_name, self.__chromosom_configs[chrom_name])
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
                new_genes = np.random.randint(low=low, high=high, size=size_needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif (size_needed < 0):
                union_genes = union_genes[:size_needed]
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
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
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
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
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
        cond = np.random.choice([0, 1, 2, 3], size=None, p=p_method)
        if   (cond == 0):
            offspring = self.crossoverSplit(parent1, parent2, low, high)
        elif (cond == 1):
            offspring = self.crossoverDiff(parent1, parent2, low, high)
        elif (cond == 2):
            offspring = self.crossoverPermSim(parent1, parent2, low, high)
        elif (cond == 3):
            offspring = self.crossoverFlipSim(parent1, parent2, low, high)
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

