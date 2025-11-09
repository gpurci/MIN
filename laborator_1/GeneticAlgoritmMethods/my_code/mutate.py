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
    def __init__(self, config):
        super().__init__()
        self.setConfig(config)

    def __call__(self, parent1, parent2, offspring):
        # calcularea ratei de probabilitate a mutatiei
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.MUTATION_RATE): # aplicarea operatiei de mutatie
            offspring = self.mutate(parent1, parent2, offspring)
        return offspring

    def __config_fn(self):
        self.mutate = self.mutateAbstract
        if (self.__config is not None):
            if   (self.__config == "inversion"):
                self.mutate = self.mutateInversion
            elif (self.__config == "scramble"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.p_mut = [1/2, 1/2]
                self.mutate = self.mutateScramble
            elif (self.__config == "swap"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.p_mut = [1/4, 3/4]
                self.mutate = self.mutateSwap
            elif (self.__config == "roll"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.mutate = self.mutateRoll

                
        else:
            pass

    def help(self):
        info = """Mutate: 
        metode de config: 'inversion', 'scramble', 'swap', 'roll', \n"""
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def mutateAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste configuratia pentru functia de 'Mutate': config '{}'".format(self.__config))

    def mutateSwap(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # cond 0 -> mutatie, metoda swap
        # cond 1 -> mutatie, este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
        cond = np.random.choice([0, 1], size=None, p=self.p_mut)# self.p_mut - se calculeaza la configurare in call
        # obtinere locus-urile aleator
        loc1, loc2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=2)
        # aplica mutatia
        if   (cond == 0):
            offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        elif (cond == 1):
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
                offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        return offspring

    def mutateRoll(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=1, high=self.GENOME_LENGTH-6, size=None)
        offspring = np.roll(offspring, size_shift)

    def mutateScramble(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-6, size=None)
        shufle_genes = np.random.permutation(offspring[size_shift:size_shift+5])
        offspring[size_shift:size_shift+5] = shufle_genes
        return offspring

    def mutateInversion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-6, size=None)
        args = np.arange(size_shift, size_shift+5)
        offspring[args] = np.flip(offspring[args])
        return offspring

    def mutateInsertion(self, parent1, parent2, offspring): # TO DO
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        size_shift = np.random.randint(low=0, high=self.GENOME_LENGTH-6, size=None)
        args = np.arange(size_shift, size_shift+5)
        offspring[args] = np.flip(offspring[args])
        return offspring
