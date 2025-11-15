#!/usr/bin/python

import numpy as np
from root_GA import *

class Mutate(RootGA):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, **chromozomes):
        super().__init__()
        self.__genome     = genome
        self.__chromozoms = chromozomes
        self.__setMethods()

    def __str__(self):
        info = "Mutate:\n"
        for chrom_name in self.__genome.keys():
            tmp   = "Chromozome name: '{}', method '{}', configs: '{}'\n".format(chrom_name, self.__methods[chrom_name], self.__chromozoms[chrom_name])
            info += "\t{}".format(tmp)
        return info

    def __call__(self, parent1, parent2, offspring):
        tmp_genome = []
        for chromozome_name in self.__genome.keys():
            tmp_genome.append(self.__call_chromozome(parent1, parent2, offspring, chromozome_name))
        return self.__genome.concat(tmp_genome)

    def __call_chromozome(self, parent1, parent2, offspring, chromozome_name):
        # calcularea ratei de probabilitate a mutatiei
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.MUTATION_RATE): # aplicarea operatiei de mutatie
            offspring = self.__fn[chromozome_name](parent1[chromozome_name], parent2[chromozome_name], 
                                                    offspring[chromozome_name], 
                                                    **self.__chromozoms[chromozome_name])
        else:
            offspring = offspring[chromozome_name]

        return offspring

    def __unpack_method(self, method):
        fn = self.mutateAbstract
        if (method is not None):
            if   (method == "inversion"):
                fn = self.mutateInversion
            elif (method == "scramble"):
                fn = self.mutateScramble
            elif (method == "swap"):
                fn = self.mutateSwap
            elif (method == "diff_swap"):
                fn = self.mutateDiffSwap
            elif (method == "roll"):
                fn = self.mutateRoll
            elif (method == "insertion"):
                fn = self.mutateInsertion
            elif (method == "rool_sim"):
                fn = self.mutateRollSim
            elif (method == "perm_sim"):
                fn = self.mutatePermSim
            elif (method == "flip_sim"):
                fn = self.mutateFlipSim
            elif (method == "binary"):
                fn = self.mutateBinary
            elif (method == "binary_sim"):
                fn = self.mutateBinarySim
            elif (method == "binary_mixt"):
                fn = self.mutateBinaryMixt
            elif (method == "mixt"):
                fn = self.mutateMixt

        return fn

    def help(self):
        info = """Mutate:
    metoda: 'inversion'; config: -> "subset_size":7;
    metoda: 'scramble';  config: -> "subset_size":7;
    metoda: 'swap';      config: None;
    metoda: 'roll';      config: None;
    metoda: 'insertion'; config: None;
    metoda: 'rool_sim';  config: None;
    metoda: 'perm_sim';  config: None;
    metoda: 'flip_sim';  config: None;
    metoda: 'binary';    config: None;
    metoda: 'binary_sim';config: None;
    metoda: 'binary_mixt';config: -> "p_method":[4/10, 1/10, 1/10, 3/10, 1/10], "subset_size":7;
    metoda: 'mixt';      config: -> "p_method":[4/10, 1/10, 1/10, 3/10, 1/10], "subset_size":7;\n"""
        return info

    def __setMethods(self):
        self.__fn      = {}
        self.__methods = {}
        for key in self.__genome.keys():
            method = self.__chromozoms[key].pop("method", None)
            self.__methods[key] = method
            self.__fn[key]      = self.__unpack_method(method)

    def mutateAbstract(self, parent1, parent2, offspring):
        error_mesage = ""
        for chrom_name in self.__genome.keys():
            error_mesage += "Lipseste metoda '{}' pentru chromozomul '{}', functia de 'Mutate': config '{}'\n".format(self.__methods[chrom_name], chrom_name, self.__chromozoms[chrom_name])
        raise NameError(error_mesage)

    def mutateSwap(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # obtinere locus-urile aleator
        loc1, loc2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=2)
        offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        return offspring

    def mutateDiffSwap(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
        este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele, unde codul genetic al parintilor este identic
        mask = parent1==parent2
        similar_locus = np.argwhere(mask).reshape(-1)
        if (similar_locus.shape[0] > 1):
            similar_locus = similar_locus.reshape(-1)
            # obtine locusul 1
            locus1        = np.random.permutation(similar_locus)[0]
            # obtine locus-urile pentru genele diferite
            not_similar_locus = np.ones(self.GENOME_LENGTH, dtype=bool)
            # obtine genele similare
            similar_genes = parent1[similar_locus]
            not_similar_locus[similar_genes] = False
            # obtine o gena de pe locusuri diferite
            not_similar_locus = np.argwhere(not_similar_locus).reshape(-1)
            if (not_similar_locus.shape[0] > 0):
                locus2 = np.random.permutation(not_similar_locus)[0]
            else:
                locus2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
            # schimba genele
            offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
        else:
            # obtinere locus-urile aleator
            locus1, locus2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=2)
            offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
        return offspring

    def mutateRoll(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=1, high=self.GENOME_LENGTH-2, size=None)
        offspring  = np.roll(offspring, size_shift)
        return offspring

    def mutateScramble(self, parent1, parent2, offspring, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-(subset_size+1), size=None)
        locuses    = np.arange(size_shift, size_shift+subset_size)
        shufle_genes = np.random.permutation(offspring[locuses])
        offspring[locuses] = shufle_genes
        return offspring

    def mutateInversion(self, parent1, parent2, offspring, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-(subset_size+1), size=None)
        locuses    = np.arange(size_shift, size_shift+subset_size)
        offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def mutateInsertion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus1 = np.random.randint(low=0,      high=self.GENOME_LENGTH//2, size=None)
        locus2 = np.random.randint(low=locus1, high=self.GENOME_LENGTH-1,  size=None)
        # copy gene
        gene1  = offspring[locus1]
        # make change locuses
        locuses= np.arange(locus1, locus2)
        offspring[locuses] = offspring[locuses+1]
        offspring[locus2]  = gene1
        return offspring

    def mutateRollSim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 3):
                locuses    = np.arange(start, start+lenght)
                size_shift = np.random.randint(low=1, high=lenght//2, size=None)
                offspring[locuses] = np.roll(offspring[locuses], size_shift)
        return offspring

    def mutatePermSim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.random.permutation(offspring[locuses])
        return offspring

    def mutateFlipSim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def mutateBinary(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH-1, size=None)
        offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateBinarySim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele care sunt asemanatoare
        mask_sim_locus = parent1==parent2
        sim_locus = np.argwhere(mask_sim_locus)
        size = sim_locus.shape[0]
        if (size >= 4):
            start, lenght = recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locus = np.random.randint(low=start, high=start+lenght, size=None)
                offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateBinaryMixt(self, parent1, parent2, offspring, p_method=None, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2, 3, 4], size=None, p=p_method)
        if   (cond == 0):
            offspring = self.mutateDiffSwap(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        elif (cond == 2):
            offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        elif (cond == 3):
            offspring = self.mutateInsertion(parent1, parent2, offspring)
        elif (cond == 4):
            offspring = self.mutateBinary(parent1, parent2, offspring)
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_method=None, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2, 3, 4], size=None, p=p_method)
        if   (cond == 0):
            offspring = self.mutateDiffSwap(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        elif (cond == 2):
            offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        elif (cond == 3):
            offspring = self.mutateInsertion(parent1, parent2, offspring)
        elif (cond == 4):
            offspring = self.mutateRollSim(parent1, parent2, offspring)
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
