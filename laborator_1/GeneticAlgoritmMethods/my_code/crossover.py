import numpy as np
from root_GA import *


class Crossover(RootGA):
    """
    Clasa 'Crossover' – metode pentru incrucisarea genetica.
    """

    def __init__(self, genome, **chromozoms):
        super().__init__()
        self.__genome = genome
        self.__chromozoms = chromozoms
        self.__setMethods()

    # -------------------------------------------------------
    def __call__(self, parent1, parent2):
        tmp_genome = []
        for chromozome_name in self.__genome.keys():
            tmp_genome.append(
                self.__call_chromozome(parent1, parent2, chromozome_name)
            )
        return self.__genome.concat(tmp_genome)

    # -------------------------------------------------------
    def __call_chromozome(self, parent1, parent2, chromozome_name):

        # probabilitate de crossover
        rate = np.random.uniform(0, 1)

        if rate <= self.CROSSOVER_RATE:
            low, high = self.__genome.getGeneRange(chromozome_name)
            offspring = self.__fn[chromozome_name](
                parent1[chromozome_name],
                parent2[chromozome_name],
                low,
                high,
                **self.__chromozoms[chromozome_name]
            )
        else:
            offspring = parent1[chromozome_name].copy()

        return offspring

    # -------------------------------------------------------
    def __str__(self):
        info = "Crossover:\n"
        for chrom_name in self.__genome.keys():
            info += (
                f"\tChromozom '{chrom_name}', "
                f"method '{self.__methods[chrom_name]}', "
                f"configs: {self.__chromozoms[chrom_name]}\n"
            )
        return info

    # -------------------------------------------------------
    def __unpack_method(self, method):

        fn = self.crossoverAbstract

        if method == "diff":
            fn = self.crossoverDiff
        elif method == "split":
            fn = self.crossoverSplit
        elif method == "perm_sim":
            fn = self.crossoverPermSim
        elif method == "flip_sim":
            fn = self.crossoverFlipSim
        elif method == "mixt":
            fn = self.crossoverMixt
        elif method == "uniform":
            fn = self.crossoverUniform
        elif method == "binary":
            fn = self.crossoverBinary

        return fn

    # -------------------------------------------------------
    def help(self):
        return """Crossover:
    method: 'diff'
    method: 'split'
    method: 'perm_sim'
    method: 'mixt', config -> p_method=[0.4,0.3,0.3]
"""

    # -------------------------------------------------------
    def __setMethods(self):
        self.__fn = {}
        self.__methods = {}

        for chrom in self.__genome.keys():
            method = self.__chromozoms[chrom].pop("method", None)
            self.__methods[chrom] = method
            self.__fn[chrom] = self.__unpack_method(method)

    # -------------------------------------------------------
    def crossoverAbstract(self, parent1, parent2, low, high):
        msg = ""
        for chrom in self.__genome.keys():
            msg += (
                f"Lipseste metoda '{self.__methods[chrom]}' pentru chromozomul '{chrom}',"
                f" functia 'Crossover': config={self.__chromozoms[chrom]}\n"
            )
        raise NameError(msg)

    # -------------------------------------------------------
    def crossoverBinary(self, parent1, parent2, low, high):
        """Two-point crossover for binary knapsack vectors."""
        L = len(parent1)
        p1, p2 = np.random.randint(0, L, size=2)
        if p1 > p2:
            p1, p2 = p2, p1
        offspring = parent1.copy()
        offspring[p1:p2] = parent2[p1:p2]
        return offspring

    # -------------------------------------------------------
    def crossoverDiff(self, parent1, parent2, low, high):
        """TSP crossover based on differing genes."""
        offspring = parent1.copy()

        mask = parent1 != parent2
        diff_locus = np.argwhere(mask)
        size = diff_locus.shape[0]

        if size >= 4:

            diff_locus = diff_locus.reshape(-1)
            diff_genes1 = parent1[diff_locus]
            diff_genes2 = parent2[diff_locus]

            union_genes = np.union1d(diff_genes1, diff_genes2)

            needed = diff_locus.shape[0] - union_genes.shape[0]

            if needed > 0:
                new_genes = np.random.randint(low=low, high=high, size=needed)
                union_genes = np.concatenate([union_genes, new_genes])
            elif needed < 0:
                union_genes = union_genes[:needed]

            union_genes = np.random.permutation(union_genes)
            offspring[diff_locus] = union_genes

        return offspring

    # -------------------------------------------------------
    def crossoverSplit(self, parent1, parent2, low, high):

        offspring = parent1.copy()

        start, end = np.random.randint(0, self.GENOME_LENGTH, size=2)
        if start > end:
            start, end = end, start

        offspring[start:end] = parent2[start:end]
        return offspring

    # -------------------------------------------------------
    def crossoverPermSim(self, parent1, parent2, low, high):

        offspring = parent1.copy()
        mask = parent1 == parent2
        size = np.sum(mask)

        if size >= 4:
            start, length = recSim(mask, 0, 0, 0)
            if length > 1:
                loc = np.arange(start, start + length)
                offspring[loc] = np.random.permutation(offspring[loc])

        return offspring

    # -------------------------------------------------------
    def crossoverFlipSim(self, parent1, parent2, low, high):

        offspring = parent1.copy()
        mask = parent1 == parent2
        size = np.sum(mask)

        if size >= 4:
            start, length = recSim(mask, 0, 0, 0)
            if length > 1:
                loc = np.arange(start, start + length)
                offspring[loc] = np.flip(offspring[loc])

        return offspring

    # -------------------------------------------------------
    def crossoverMixt(self, parent1, parent2, low, high, p_method=None):

        cond = np.random.choice([0, 1, 2], p=p_method)

        if cond == 0:
            return self.crossoverSplit(parent1, parent2, low, high)
        elif cond == 1:
            return self.crossoverPermSim(parent1, parent2, low, high)
        else:
            return self.crossoverFlipSim(parent1, parent2, low, high)

    # -------------------------------------------------------
    def crossoverUniform(self, parent1, parent2, low, high):
        mask = np.random.rand(len(parent1)) < 0.5
        return np.where(mask, parent1, parent2)


# -----------------------------------------------------------
def recSim(mask_genes, start, length, arg):
    """Returneaza cea mai lunga secventa continua unde genele sunt identice."""
    if arg < mask_genes.shape[0]:
        tmp_arg = arg
        tmp_st = arg
        tmp_len = 0
        while tmp_arg < mask_genes.shape[0]:
            if mask_genes[tmp_arg]:
                tmp_arg += 1
            else:
                tmp_len = tmp_arg - tmp_st
                if length < tmp_len:
                    start, length = tmp_st, tmp_len
                return recSim(mask_genes, start, length, tmp_arg + 1)
    return start, length
