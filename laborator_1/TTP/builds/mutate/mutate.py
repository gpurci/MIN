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
            if   (method == "swap"):
                fn = self.mutateSwap
            elif (method == "swap_sim"):
                fn = self.mutateSwapSim
            elif (method == "swap_diff"):
                fn = self.mutateSwapDiff

            elif (method == "roll"):
                fn = self.mutateRoll
            elif (method == "rool_sim"):
                fn = self.mutateRollSim
            elif (method == "rool_diff"):
                fn = self.mutateRollDiff

            elif (method == "scramble"):
                fn = self.mutateScramble
            elif (method == "scramble_sim"):
                fn = self.mutateScrambleSim
            elif (method == "scramble_diff"):
                fn = self.mutateScrambleDiff

            elif (method == "inversion"):
                fn = self.mutateInversion
            elif (method == "inversion_sim"):
                fn = self.mutateInversionSim
            elif (method == "inversion_diff"):
                fn = self.mutateInversionDiff

            elif (method == "insertion"):
                fn = self.mutateInsertion

            elif (method == "mixt"):
                fn = self.mutateMixt

        return fn

    def help(self):
        info = """Mutate:
    metoda: 'swap';      config: None;
    metoda: 'swap_sim';  config: None;
    metoda: 'swap_diff'; config: None;
    metoda: 'roll';      config: -> "subset_size":7;
    metoda: 'rool_sim';  config: -> "subset_size":7;
    metoda: 'rool_diff'; config: -> "subset_size":7;
    metoda: 'scramble';  config: -> "subset_size":7;
    metoda: 'scramble_sim';   config: None;
    metoda: 'scramble_diff';  config: None;
    metoda: 'inversion';      config: -> "subset_size":7;
    metoda: 'inversion_sim';  config: None;
    metoda: 'inversion_diff'; config: None;
    metoda: 'insertion'; config: None;
    metoda: 'mixt';      config: -> "p_method":[1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13], "subset_size":7;\n"""
        print(info)

    def mutateAbstract(self, parent1, parent2, offspring):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
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

    def mutateSwapSim(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
        este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele, unde codul genetic al parintilor este identic
        mask_locus = parent1==parent2
        similar_locus  = np.argwhere(mask_locus).reshape(-1)
        if (similar_locus.shape[0] > 1):
            similar_locus = similar_locus.reshape(-1)
            # obtine locusul 1
            locus1        = np.random.permutation(similar_locus)[0]
            # obtine locus-urile pentru genele diferite
            not_similar_locus = np.invert(mask_locus)
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
            offspring = self.mutateSwap(parent1, parent2, offspring)
        return offspring

    def mutateSwapDiff(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
        este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # modifica doar genele, unde codul genetic al parintilor este identic
        mask_locus = parent1!=parent2
        diff_locus  = np.argwhere(mask_locus).reshape(-1)
        if (diff_locus.shape[0] > 1):
            diff_locus = diff_locus.reshape(-1)
            # obtine locusul 1
            locus1     = np.random.permutation(diff_locus)[0]
            # obtine locus-urile pentru genele diferite
            similar_locus = np.invert(mask_locus)
            # obtine o gena de pe locusuri diferite
            similar_locus = np.argwhere(similar_locus).reshape(-1)
            if (similar_locus.shape[0] > 0):
                locus2 = np.random.permutation(similar_locus)[0]
            else:
                locus2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
            # schimba genele
            offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
        else:
            # obtinere locus-urile aleator
            offspring = self.mutateSwap(parent1, parent2, offspring)
        return offspring

    def __mutateRoll(self, offspring, start, end, subset_size):
        size_shift = np.random.randint(low=1, high=subset_size, size=None)
        # gaseste locusul unde vom modifica
        locuses = np.arange(start, end)
        # aplica deplasarea
        offspring[locuses] = np.roll(offspring[locuses], size_shift)
        return offspring

    def mutateRoll(self, parent1, parent2, offspring, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # gaseste locusul unde vom modifica
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        return self.__mutateRoll(offspring, locus, locus+subset_size, subset_size)

    def mutateRollSim(self, parent1, parent2, offspring, subset_size=7):
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
            if (lenght > 2):
                offspring = self.__mutateRoll(offspring, start, start+lenght, lenght)
            else:
                offspring = self.mutateRoll(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateRoll(parent1, parent2, offspring, subset_size)

        return offspring

    def mutateRollDiff(self, parent1, parent2, offspring, subset_size=7):
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
            if (lenght > 2):
                offspring = self.__mutateRoll(offspring, start, start+lenght, lenght)
            else:
                offspring = self.mutateRoll(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateRoll(parent1, parent2, offspring, subset_size)
        return offspring

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
        locus   = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
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

    def mutateInversion(self, parent1, parent2, offspring, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus   = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        locuses = np.arange(locus, locus+subset_size)
        offspring[locuses] = np.flip(offspring[locuses])
        return offspring

    def mutateInversionSim(self, parent1, parent2, offspring, subset_size=7):
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
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
            else:
                offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        return offspring

    def mutateInversionDiff(self, parent1, parent2, offspring, subset_size=7):
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
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
            else:
                offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        else:
            offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        return offspring

    def mutateInsertion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus1 = np.random.randint(low=0,      high=self.GENOME_LENGTH//2, size=None)
        locus2 = np.random.randint(low=locus1, high=self.GENOME_LENGTH,    size=None)
        # copy gene
        gene1  = offspring[locus1]
        # make change locuses
        locuses= np.arange(locus1, locus2)
        offspring[locuses] = offspring[locuses+1]
        offspring[locus2]  = gene1
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_method=None, subset_size=7):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], size=None, p=p_method)
        # swap
        if   (cond == 0):
            offspring = self.mutateSwap(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateSwapSim(parent1, parent2, offspring)
        elif (cond == 2):
            offspring = self.mutateSwapDiff(parent1, parent2, offspring)
        # roll
        elif (cond == 3):
            offspring = self.mutateRoll(parent1, parent2, offspring, subset_size)
        elif (cond == 4):
            offspring = self.mutateRollSim(parent1, parent2, offspring, subset_size)
        elif (cond == 5):
            offspring = self.mutateRollDiff(parent1, parent2, offspring, subset_size)
        # scramble
        elif (cond == 6):
            offspring = self.mutateScramble(parent1, parent2, offspring, subset_size)
        elif (cond == 7):
            offspring = self.mutateScrambleSim(parent1, parent2, offspring, subset_size)
        elif (cond == 8):
            offspring = self.mutateScrambleDiff(parent1, parent2, offspring, subset_size)
        # inversion
        elif (cond == 9):
            offspring = self.mutateInversion(parent1, parent2, offspring, subset_size)
        elif (cond == 10):
            offspring = self.mutateInversionSim(parent1, parent2, offspring)
        elif (cond == 11):
            offspring = self.mutateInversionDiff(parent1, parent2, offspring)
        # insertion
        elif (cond == 12):
            offspring = self.mutateInsertion(parent1, parent2, offspring)
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
