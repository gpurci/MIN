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
    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)
        self.__subset_size = 2

    def __call__(self, parent1, parent2, offspring):
        # calcularea ratei de probabilitate a mutatiei
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.MUTATION_RATE): # aplicarea operatiei de mutatie
            offspring = self.fn(parent1, parent2, offspring)
        return offspring

    def __method_fn(self):
        self.fn = self.mutateAbstract
        if (self.__method is not None):
            if   (self.__method == "inversion"):
                self.fn = self.mutateInversion
            elif (self.__method == "scramble"):
                self.fn = self.mutateScramble
            elif (self.__method == "swap"):
                self.fn = self.mutateSwap
            elif (self.__method == "diff_swap"):
                self.fn = self.mutateDiffSwap
            elif (self.__method == "roll"):
                self.fn = self.mutateRoll
            elif (self.__method == "insertion"):
                self.fn = self.mutateInsertion
            elif (self.__method == "rool_sim"):
                self.fn = self.mutateRollSim
            elif (self.__method == "perm_sim"):
                self.fn = self.mutatePermSim
            elif (self.__method == "flip_sim"):
                self.fn = self.mutateFlipSim
            elif (self.__method == "mixt"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.__p_method = [4/10, 1/10, 1/10, 3/10, 1/10]
                self.fn = self.mutateMixtDSSII

                
        else:
            pass

    def help(self):
        info = """Mutate:
        \tmetoda: 'inversion'; config: None;
        \tmetoda: 'scramble';  config: None;
        \tmetoda: 'swap';      config: None;
        \tmetoda: 'roll';      config: None;
        \tmetoda: 'insertion'; config: None;
        \tmetoda: 'rool_sim';  config: None;
        \tmetoda: 'perm_sim';  config: None;
        \tmetoda: 'flip_sim';  config: None;
        \tmetoda: 'mixt';      config: -> p_method[4/10, 1/10, 1/10, 3/10, 1/10], size_subset;\n"""
        return info

    def increaseSubsetSize(self):
        self.__subset_size += 1
        if (self.__subset_size > 10):
            self.__subset_size = 10

    def decreaseSubsetSize(self):
        self.__subset_size -= 1
        if (self.__subset_size < 2):
            self.__subset_size = 2

    def __setMethods(self, method):
        self.__method = method
        self.__method_fn()

    def mutateAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste metoda '{}' pentru functia de 'Mutate': config '{}'".format(self.__method, self.__config))

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

    def mutateScramble(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-(self.__subset_size+1), size=None)
        locuses    = np.arange(size_shift, size_shift+self.__subset_size)
        shufle_genes = np.random.permutation(offspring[locuses])
        offspring[locuses] = shufle_genes
        return offspring

    def mutateInversion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-(self.__subset_size+1), size=None)
        locuses    = np.arange(size_shift, size_shift+self.__subset_size)
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
            start, lenght = Mutate.recSim(mask_sim_locus, 0, 0, 0)
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
            start, lenght = Mutate.recSim(mask_sim_locus, 0, 0, 0)
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
            start, lenght = Mutate.recSim(mask_sim_locus, 0, 0, 0)
            if (lenght > 1):
                locuses = np.arange(start, start+lenght)
                offspring[locuses] = np.flip(offspring[locuses])
        return offspring


    def mutateMixtDSSII(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1, 2, 3, 4], size=None, p=self.p_mixt)
        if   (cond == 0):
            offspring = self.mutateDiffSwap(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateScramble(parent1, parent2, offspring)
        elif (cond == 2):
            offspring = self.mutateInversion(parent1, parent2, offspring)
        elif (cond == 3):
            offspring = self.mutateInsertion(parent1, parent2, offspring)
        elif (cond == 4):
            offspring = self.mutateRollSim(parent1, parent2, offspring)
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
                    return Mutate.recSim(individ, start, lenght, tmp_arg+1)
        else:
            return start, lenght
        return start, lenght

    # helper pentru debugging mutatii: TO DO: Adauga in test aici esto doar functionalul!!!
    def _diff(self, before, after):
        """returneaza tuple(index, val_before, val_after) pentru gene schimbate"""
        idx = np.where(before != after)[0]
        return [(int(i), int(before[i]), int(after[i])) for i in idx]
