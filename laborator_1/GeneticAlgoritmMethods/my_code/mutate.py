#!/usr/bin/python
import numpy as np
from root_GA import *

class Mutate(RootGA):
    """
    Clasa 'Mutate' – operatori genetici de mutație pentru TSP / TTP.
    """

    def __init__(self, genome, **chromozomes):
        super().__init__()
        self.__genome = genome
        self.__chromozoms = chromozomes
        self.__setMethods()

    def __str__(self):
        info = "Mutate:\n"
        for chrom_name in self.__genome.keys():
            tmp = (
                f"Chromozome name: '{chrom_name}', "
                f"method '{self.__methods[chrom_name]}', "
                f"configs: '{self.__chromozoms[chrom_name]}'\n"
            )
            info += f"\t{tmp}"
        return info

    # ===============================================================
    # Main __call__
    # ===============================================================
    def __call__(self, parent1, parent2, offspring):
        tmp_genome = []
        for name in self.__genome.keys():
            tmp_genome.append(
                self.__call_chromozome(parent1, parent2, offspring, name)
            )
        return self.__genome.concat(tmp_genome)

    def __call_chromozome(self, parent1, parent2, offspring, chrom_name):
        rate = np.random.uniform(0, 1)
        if rate <= self.MUTATION_RATE:
            return self.__fn[chrom_name](
                parent1[chrom_name],
                parent2[chrom_name],
                offspring[chrom_name],
                **self.__chromozoms[chrom_name]
            )
        return offspring[chrom_name]

    # ===============================================================
    # Method selection
    # ===============================================================
    def __setMethods(self):
        self.__methods = {}
        self.__fn = {}

        for key in self.__genome.keys():
            method = self.__chromozoms[key].pop("method", None)
            self.__methods[key] = method
            self.__fn[key] = self.__unpack_method(method)

    def __unpack_method(self, method):
        # default
        fn = self.mutateAbstract

        if method is None:
            return fn

        return {
            "segment_invert": self.mutateInvertSegment,
            "inversion": self.mutateInversion,
            "scramble": self.mutateScramble,
            "swap": self.mutateSwap,
            "diff_swap": self.mutateDiffSwap,
            "roll": self.mutateRoll,
            "insertion": self.mutateInsertion,
            "rool_sim": self.mutateRollSim,
            "perm_sim": self.mutatePermSim,
            "flip_sim": self.mutateFlipSim,
            "binary": self.mutateBinary,
            "binary_sim": self.mutateBinarySim,
            "binary_mixt": self.mutateBinaryMixt,
            "mixt": self.mutateMixt,
            "bitflip": self.mutateBitflip,
            "mixt_ttp": self.mutateMixtTTP,
            "hybrid_2opt": self.mutateHybrid2Opt,
        }.get(method, fn)

    def help(self):
        return (
            "Mutate: metoda: 'inversion'; config: subset_size\n"
            "        metoda: 'scramble'; config: subset_size\n"
            "        metoda: 'swap';\n"
            "        metoda: 'roll';\n"
            "        metoda: 'insertion';\n"
            "        metoda: 'binary';\n"
            "        metoda: 'bitflip';\n"
            "        metoda: 'mixt_ttp';\n"
        )

    # ===============================================================
    # FAILBACK
    # ===============================================================
    def mutateAbstract(self, parent1, parent2, offspring):
        msg = ""
        for chrom in self.__genome.keys():
            msg += (
                f"Lipseste metoda '{self.__methods[chrom]}' pentru "
                f"chromozomul '{chrom}', config '{self.__chromozoms[chrom]}'\n"
            )
        raise NameError(msg)

    # ===============================================================
    # Basic Operators
    # ===============================================================

    def mutateBitflip(self, parent1, parent2, offspring, rate=0.01):
        """Flips each bit with probability `rate`."""
        offspring = offspring.copy()
        mask = np.random.rand(len(offspring)) < rate
        offspring[mask] = 1 - offspring[mask]
        return offspring

    def mutateSwap(self, parent1, parent2, offspring):
        loc1, loc2 = np.random.randint(0, self.GENOME_LENGTH, size=2)
        offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        return offspring
    
    def invert_segment(self, tour):
        a, b = sorted(np.random.choice(len(tour), 2, replace=False))
        tour[a:b] = tour[a:b][::-1]
        return tour
    
    def mutateInvertSegment(self, parent1, parent2, offspring, **kw):
        """Wrapper for invert_segment so GA can call it."""
        return self.invert_segment(offspring.copy())


    def mutateDiffSwap(self, parent1, parent2, offspring):
        mask = parent1 == parent2
        loci = np.argwhere(mask).reshape(-1)

        if len(loci) > 1:
            locus1 = np.random.choice(loci)

            diff_mask = np.ones(self.GENOME_LENGTH, dtype=bool)
            diff_mask[parent1[loci]] = False
            diff_loci = np.argwhere(diff_mask).reshape(-1)

            if len(diff_loci) > 0:
                locus2 = np.random.choice(diff_loci)
            else:
                locus2 = np.random.randint(0, self.GENOME_LENGTH)
        else:
            locus1, locus2 = np.random.randint(0, self.GENOME_LENGTH, size=2)

        offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
        return offspring

    def mutateRoll(self, parent1, parent2, offspring):
        shift = np.random.randint(1, self.GENOME_LENGTH - 1)
        return np.roll(offspring, shift)

    def mutateScramble(self, parent1, parent2, offspring, subset_size=7):
        start = np.random.randint(0, self.GENOME_LENGTH - subset_size)
        loc = slice(start, start + subset_size)
        offspring[loc] = np.random.permutation(offspring[loc])
        return offspring

    def mutateInversion(self, parent1, parent2, offspring, subset_size=7):
        start = np.random.randint(0, self.GENOME_LENGTH - subset_size)
        loc = slice(start, start + subset_size)
        offspring[loc] = offspring[loc][::-1]
        return offspring

    def mutateInsertion(self, parent1, parent2, offspring):
        i = np.random.randint(0, self.GENOME_LENGTH // 2)
        j = np.random.randint(i, self.GENOME_LENGTH - 1)
        gene = offspring[i]
        offspring[i:j] = offspring[i + 1 : j + 1]
        offspring[j] = gene
        return offspring

    # ===============================================================
    # Similarity-based Operators
    # ===============================================================
    def mutateRollSim(self, parent1, parent2, offspring):
        mask = parent1 == parent2
        start, length = recSim(mask, 0, 0, 0)

        if length > 3:
            loc = slice(start, start + length)
            shift = np.random.randint(1, length // 2)
            offspring[loc] = np.roll(offspring[loc], shift)

        return offspring

    def mutatePermSim(self, parent1, parent2, offspring):
        mask = parent1 == parent2
        start, length = recSim(mask, 0, 0, 0)

        if length > 1:
            loc = slice(start, start + length)
            offspring[loc] = np.random.permutation(offspring[loc])

        return offspring

    def mutateFlipSim(self, parent1, parent2, offspring):
        mask = parent1 == parent2
        start, length = recSim(mask, 0, 0, 0)

        if length > 1:
            loc = slice(start, start + length)
            offspring[loc] = offspring[loc][::-1]

        return offspring

    def mutateBinary(self, parent1, parent2, offspring):
        locus = np.random.randint(0, self.GENOME_LENGTH)
        offspring[locus] ^= 1
        return offspring

    def mutateBinarySim(self, parent1, parent2, offspring):
        mask = parent1 == parent2
        start, length = recSim(mask, 0, 0, 0)

        if length > 1:
            locus = np.random.randint(start, start + length)
            offspring[locus] ^= 1

        return offspring

    def mutateBinaryMixt(self, parent1, parent2, offspring, p_method=None, subset_size=7):
        op = np.random.choice([0, 1, 2, 3, 4], p=p_method)

        if op == 0:
            return self.mutateDiffSwap(parent1, parent2, offspring)
        if op == 1:
            return self.mutateScramble(parent1, parent2, offspring, subset_size)
        if op == 2:
            return self.mutateInversion(parent1, parent2, offspring, subset_size)
        if op == 3:
            return self.mutateInsertion(parent1, parent2, offspring)
        if op == 4:
            return self.mutateBinary(parent1, parent2, offspring)

        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_method=None, subset_size=7):
        op = np.random.choice([0, 1, 2, 3, 4], p=p_method)

        if op == 0:
            return self.mutateDiffSwap(parent1, parent2, offspring)
        if op == 1:
            return self.mutateScramble(parent1, parent2, offspring, subset_size)
        if op == 2:
            return self.mutateInversion(parent1, parent2, offspring, subset_size)
        if op == 3:
            return self.mutateInsertion(parent1, parent2, offspring)
        if op == 4:
            return self.mutateRollSim(parent1, parent2, offspring)

        return offspring

    # ===============================================================
    # HYBRID 2-OPT (TTP-AWARE)
    # ===============================================================
    def mutateHybrid2Opt(
        self,
        parent1,
        parent2,
        offspring,
        kp_bits=None,
        dataset=None,
        alpha=0.01,
        v_min=0.1,
        v_max=1.0,
        W=25936,
        R=5.61,
        max_samples=8,
    ):
        if kp_bits is None or dataset is None:
            return offspring

        route = offspring.copy()
        n = len(route) - 1

        distance = dataset["distance"]
        item_profit = dataset["item_profit"]
        item_weight = dataset["item_weight"]

        def ttp_score(r):
            Pcur = 0.0
            Wcur = 0.0
            Tcur = 0.0

            for i in range(len(r) - 1):
                city = r[i]

                if city > 0 and kp_bits[city] == 1:
                    item_idx = city - 1
                    profit = item_profit[item_idx]
                    weight = item_weight[item_idx]

                    Pcur += max(0, profit - alpha * Tcur)
                    Wcur += weight

                v = max(v_min, v_max - (v_max - v_min) * (Wcur / W))
                nxt = r[i + 1]
                Tcur += distance[city, nxt] / v

            last = r[-1]
            start = r[0]
            v = max(v_min, v_max - (v_max - v_min) * (Wcur / W))
            Tcur += distance[last, start] / v

            return Pcur - R * Tcur

        base = ttp_score(route)

        for _ in range(max_samples):
            i = np.random.randint(1, n - 1)
            k = np.random.randint(i + 1, n)

            new_route = route.copy()
            new_route[i : k + 1] = new_route[i : k + 1][::-1]

            new_score = ttp_score(new_route)

            if new_score > base:
                route = new_route
                base = new_score

        return route

    # ===============================================================
    # MIXT TTP (COMBINED TSP + KP CLEANUP)
    # ===============================================================
    def mutateMixtTTP(
        self,
        parent1,
        parent2,
        offspring,
        p_method=None,
        subset_size=20,
        kp_bits=None,
        dataset=None,
        alpha=0.01,
        v_min=0.1,
        v_max=1.0,
        W=25936,
        R=5.61,
    ):
        if kp_bits is None or dataset is None or p_method is None:
            return offspring

        route = offspring.copy()

        # -------- hybrid 2-opt shortcut --------
        def apply_h2opt(r):
            return self.mutateHybrid2Opt(
                parent1,
                parent2,
                r,
                kp_bits=kp_bits,
                dataset=dataset,
                alpha=alpha,
                v_min=v_min,
                v_max=v_max,
                W=W,
                R=R,
            )

        def tsp_micro_op(r):
            op = np.random.choice([0, 1, 2, 3, 4], p=p_method)
            if op == 0:
                return self.mutateDiffSwap(parent1, parent2, r)
            if op == 1:
                return self.mutateInversion(parent1, parent2, r, subset_size)
            if op == 2:
                return self.mutateInsertion(parent1, parent2, r)
            if op == 3:
                return apply_h2opt(r)
            if op == 4:
                return self.invert_segment(r.copy())
            return r

        # multiple micro-operators
        n_ops = 1
        for _ in range(n_ops):
            route = tsp_micro_op(route)

        # KP cleanup (repair overweight)
        def cleanup(bits):
            profit = dataset["item_profit"][1:]
            weight = dataset["item_weight"][1:]

            mask = bits[1 : 1 + len(weight)].astype(bool)
            total = np.dot(weight, mask)

            capacity = self.__chromozoms["tsp"].get("W", 25936)

            if total <= capacity:
                return bits

            pw = np.zeros_like(profit, float)
            nonzero = weight > 0
            pw[nonzero] = profit[nonzero] / weight[nonzero]
            pw[~nonzero] = np.inf

            order = np.argsort(pw)

            for idx in order:
                city = idx + 1
                if bits[city] == 1:
                    bits[city] = 0
                    total -= weight[idx]
                    if total <= capacity:
                        break

            return bits

        new_bits = kp_bits.copy()
        cleanup(new_bits)
        return route



# ===============================================================
# Similarity helper
# ===============================================================
def recSim(mask_genes, start, length, arg):
    """Caută cea mai lungă secvență de gene identice."""
    if arg < mask_genes.shape[0]:
        tmp_arg = arg
        tmp_start = arg
        tmp_len = 0

        while tmp_arg < mask_genes.shape[0]:
            if mask_genes[tmp_arg]:
                tmp_arg += 1
            else:
                tmp_len = tmp_arg - tmp_start
                if length < tmp_len:
                    start, length = tmp_start, tmp_len
                return recSim(mask_genes, start, length, tmp_arg + 1)
    else:
        return start, length

    return start, length
