#!/usr/bin/python
import numpy as np
import warnings


class Genoms(object):
    """
    Clasa 'Genoms' – gestionează toată populația.

    • reține definirea cromozomilor (tsp, kp, etc.)
    • stochează generația curentă
    • construiește generația nouă prin append() + save()
    • oferă indexing: self[i], self[args], self.chromozomes("tsp"), etc.
    """

    def __init__(self, size=10, **gene_range):
        """
        size      = GENOME_LENGTH (dimensiunea fiecărui cromozom)
        gene_range: dict
            ex: {"tsp": (0, 280), "kp": (0, 2)}

        Valoarea tuple-ului = (min_val, max_val) pentru gene.
        """
        if not isinstance(gene_range, dict):
            raise TypeError(
                f"Parametrul 'gene_range' trebuie sa fie dict, nu {type(gene_range)}"
            )

        self.__size = size
        self.__gene_range = gene_range
        self.__keys = list(gene_range.keys())

        if len(self.__keys) == 0:
            warnings.warn(f"Lipsesc numele cromozomilor: '{gene_range}'")

        # populația curentă: nume → matrice NxGENOME_LENGTH
        self.__current = {
            name: np.zeros((0, size), dtype=np.int32)
            for name in self.__keys
        }

        # generația nouă (listă de indivizi dict)
        self.__new = []
        self.__population_size = 0

    # ------------------------------------------------------------------
    #                       STRUCTURĂ / INFO
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return (
            self.__population_size,
            len(self.__keys),
            tuple([self.__size for _ in self.__keys])
        )

    def __str__(self):
        info = "Genoms:\n"
        for key in self.__keys:
            lo, hi = self.__gene_range[key]
            info += f"\tChromozom '{key}': range ({lo}..{hi}), len={self.__size}\n"
        return info

    # ------------------------------------------------------------------
    def setSize(self, size):
        """Setează GENOME_LENGTH și resetează populația."""
        self.__size = size
        self.__current = {
            name: np.zeros((0, size), dtype=np.int32)
            for name in self.__keys
        }
        self.__new = []
        self.__population_size = 0

    def setPopulationSize(self, size):
        """Nu prealocăm nimic – populatia se formează în initPopulation."""
        pass

    def is_genoms(self):
        return self.__population_size > 0

    def keys(self):
        return self.__keys

    def getGeneRange(self, key):
        return self.__gene_range[key]

    def chromozomes(self, name):
        return self.__current[name]

    # ------------------------------------------------------------------
    #                     INDEXARE INDIVIZI
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        """
        Returnează:
          • int  -> dict cu cromozomi 1D
          • lista/array -> dict cu cromozomi 2D
        """
        out = {}
        for cname in self.__keys:
            out[cname] = self.__current[cname][key]
        return out

    def __setitem__(self, key, indiv_dict):
        """genoms[idx] = { 'tsp':..., 'kp':... }"""
        for cname in self.__keys:
            self.__current[cname][key] = indiv_dict[cname]

    # ------------------------------------------------------------------
    #                   GENERARE & SALVARE GENERAȚIE
    # ------------------------------------------------------------------
    def concat(self, chromozome_list):
        """
        Crossover/Mutate trimit listă [tsp_vec, kp_vec].
        O convertim în dict {"tsp":..., "kp":...}.
        """
        off = {}
        for cname, arr in zip(self.__keys, chromozome_list):
            off[cname] = np.array(arr, copy=True)
        return off

    def append(self, offspring):
        """Adaugă un individ dict în generația nouă."""
        for cname in self.__keys:
            if offspring[cname].shape[0] != self.__size:
                raise ValueError(
                    f"Offspring chromosome '{cname}' are length "
                    f"{offspring[cname].shape[0]}, expected {self.__size}"
                )
        self.__new.append(offspring)

    def add(self, **kw):
        """Compatibil cu codul vechi: add(tsp=..., kp=...)."""
        self.__new.append(kw)

    def save(self):
        """Finalizează generația nouă → curentă."""
        if len(self.__new) == 0:
            return

        n = len(self.__new)
        new_current = {
            cname: np.zeros((n, self.__size), dtype=np.int32)
            for cname in self.__keys
        }

        for i, indiv in enumerate(self.__new):
            for cname in self.__keys:
                new_current[cname][i] = indiv[cname]

        self.__current = new_current
        self.__population_size = n
        self.__new = []

    # ------------------------------------------------------------------
    def population(self):
        """Generator peste indivizi."""
        for i in range(self.__population_size):
            yield self[i]

    # ------------------------------------------------------------------
    def help(self):
        return 'Genoms: "name": (min_gene, max_gene), ...\n'
