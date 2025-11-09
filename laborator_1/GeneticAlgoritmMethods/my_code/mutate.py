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
        return self.mutate(parent1, parent2, offspring)

    def __config_fn(self):
        self.mutate = self.mutateAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                self.mutate = self.testParentClass
            elif (self.__config == "vecin"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.p_mut = [1-self.MUTATION_RATE, self.MUTATION_RATE/2, self.MUTATION_RATE/2]
                self.mutate = self.mutateNeighbors
            elif (self.__config == "swap"):
                # prababilitatea pentru fiecare metoda de mutatie
                self.p_mut = [1-self.MUTATION_RATE, self.MUTATION_RATE/4, 3*self.MUTATION_RATE/4]
                self.mutate = self.mutateSwap
        else:
            pass

    def help(self):
        info = """Mutate: 
        metode de config: 'vecin', 'swap'\n"""
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def mutateAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste configuratia pentru functia de 'Mutate': config '{}'".format(self.__config))

    def testParentClass(self, parent1, parent2, offspring):
        print("Mutate, testParentClass GENOME_LENGTH :{}".format(self.GENOME_LENGTH))

    def mutateNeighbors(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        raise NameError("Functia de 'Mutatie', incompleta") # TO DO
        # probabilitatea pentru fiecare metoda de mutatie
        p = [1-self.MUTATION_RATE, self.MUTATION_RATE/2, self.MUTATION_RATE/2]
        # cond 0 -> nu se aplica operatia de mutatie, descendentul ramane fara modificari
        # cond 1 -> mutatie, metoda vecinul apropiat
        # cond 2 -> mutatie, este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
        cond = np.random.choice([0, 1, 2], size=nbr_individs, p=p)
        # aplica mutatie
        if   (cond == 0):
            pass
        elif (cond == 1):
            # metoda cel mai apropiat vecin
            # 1 obtine locus-ul (loc)
            # 2 gena conditionala, care cauta cel mai apropiat vecin este alela precedenta (loc-1)
            # 3 seteaza gena de pe (loc)
            # 4 incrementeaza loc, repeta punctul (2, 3) break

            # loc - alela unde va fi aplicata mutatia, cuprinsa intre 0...GENOME_LENGTH
            loc = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
            # cond_gene - gena dupa care se va cauta cel mai apropiat vecin
            cond_genes     = individ[[loc-1, loc+1]]
            neighbors_gene = self.getNeighbors(cond_genes)
            new_gene       = np.random.permutation(neighbors_gene)[0]
            gene           = individ[loc]
            loc_new        = individ == new_gene
            individ[loc]   = new_gene
            individ[loc_new] = gene
        elif (cond == 2):
            # modifica doar genele, unde codul genetic al parintilor este identic
            mask = parent1==parent2
            mask[[0, -1]] = False # pastreaza orasul de start
            similar_locus  = np.argwhere(mask).reshape(-1)
            if (args_similar.shape[0] > 1):
                args_similar = args_similar.reshape(-1)
                # obtine genele similare
                similar_genes = parent1[args_similar]
                # sterge genele care au fost gasite
                mask_valid = np.ones(self.GENOME_LENGTH, dtype=bool)
                mask_valid[similar_genes] = False
                # adauga alte gene
                new_gene = np.argwhere(mask_valid).reshape(-1)
                new_gene = np.random.permutation(new_gene)[:2]
                args     = np.random.permutation(args_similar)[:2]
                individ[args] = new_gene

        return individs

    def mutateSwap(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # cond 0 -> nu se aplica operatia de mutatie
        # cond 1 -> mutatie, metoda swap
        # cond 2 -> mutatie, este aplicata mutatia doar pentru zonele unde codul genetic al parintilor este identic
        cond = np.random.choice([0, 1, 2], size=None, p=self.p_mut)# self.p_mut - se calculeaza la configurare in call
        # obtinere locus-urile aleator
        loc1, loc2 = np.random.randint(low=0, high=self.GENOME_LENGTH, size=2)
        # aplica mutatia
        if cond == 0:
            pass
        elif cond == 1:
            offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        elif cond == 2:
            # modifica doar genele, unde codul genetic al parintilor este identic
            mask = parent1==parent2
            similar_locus = np.argwhere(mask).reshape(-1)
            if similar_locus.shape[0] > 1:
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
                locus2            = np.random.permutation(not_similar_locus)[0]
                # schimba genele
                offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
            else:
                offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        return offspring

