#!/usr/bin/python
import numpy as np
from root_GA import *


class SelectParent(RootGA):
    """
    Clasa 'SelectParent' — metode pentru selectarea parintelui 1 sau 2.

    Folosire:
        - La începutul fiecărei generații se apelează startEpoch(fitness_values)
        - __call__() returnează indexul parintelui selectat
    """

    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.parent_arg = 0
        self.__setMethods(method)

    def __str__(self):
        return f"SelectParent: method: {self.__method} configs: {self.__configs}"

    def __call__(self):
        return self.fn(**self.__configs)

    # -------------------------------------------------------
    #      SELECT METHOD BASED ON CONFIGURATION
    # -------------------------------------------------------
    def __unpack_method(self, method):
        fn = self.selectParentAbstract
        if method is not None:
            if method == "choice":
                fn = self.selectParentChoice
            elif method == "wheel":
                fn = self.selectParentWheel
            elif method == "tour":
                fn = self.selectParentTour
            elif method == "tour_choice":
                fn = self.selectParentTourChoice
            elif method == "rise":
                fn = self.selectParentRise
            elif method == "mixt":
                fn = self.selectParentMixt
        return fn

    def help(self):
        return (
            "SelectParent:\n"
            "   metoda: 'choice'; config: None\n"
            "   metoda: 'wheel'; config: None\n"
            "   metoda: 'tour'; config: -> size_subset:7\n"
            "   metoda: 'tour_choice'; config: -> size_subset:7\n"
            "   metoda: 'rise'; config: None\n"
            "   metoda: 'mixt'; config: -> p_select=[1/4,1/4,1/4,1/4], size_subset:7\n"
        )

    def __setMethods(self, method):
        self.__method = method
        self.fn = self.__unpack_method(method)

    # -------------------------------------------------------
    #      PREPARE FOR SELECTION IN THIS GENERATION
    # -------------------------------------------------------
    def startEpoch(self, fitness_values):
        total_fitness = fitness_values.sum()

        if total_fitness != 0:
            # selectare subset pentru competitie
            size = int(self.SELECT_RATE * self.POPULATION_SIZE)

            # selectăm cei mai slabi → îi eliminăm
            args_weaks = np.argpartition(fitness_values, size)[:size]
            fitness_values[args_weaks] = 0.0

            total_fitness = fitness_values.sum()
            self.fitness_values = fitness_values / total_fitness

        else:
            # fallback: toți au aceeași șansă
            self.fitness_values = np.full(
                fitness_values.shape[0],
                1.0 / self.POPULATION_SIZE,
                dtype=np.float32
            )

    # -------------------------------------------------------
    #                 SELECTION METHODS
    # -------------------------------------------------------
    def selectParentAbstract(self, **kw):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru functia de 'SelectionParent': "
            f"config '{self.__configs}'"
        )

    def selectParentChoice(self):
        """Selectie directă pe baza probabilităților fitness."""
        return np.random.choice(
            self.POPULATION_SIZE,
            p=self.fitness_values
        )

    def selectParentWheel(self):
        """Selecție prin roata norocului."""
        pick = np.random.uniform(0, 1)
        current = 0.0

        for arg, f in enumerate(self.fitness_values):
            current += f
            if current > pick:
                return arg

        # fallback (nu ar trebui să se întâmple)
        return self.POPULATION_SIZE - 1

    def selectParentTour(self, size_subset=7):
        """Tournament selection: random subset + alegem cel mai bun."""
        args_k = np.random.randint(0, self.POPULATION_SIZE, size_subset)
        best = np.argmax(self.fitness_values[args_k])
        return args_k[best]

    def selectParentTourChoice(self, size_subset=7):
        """Tournament cu alegere ponderată."""
        args_k = np.random.choice(
            self.POPULATION_SIZE,
            size=size_subset,
            p=self.fitness_values
        )
        best = np.argmax(self.fitness_values[args_k])
        return args_k[best]

    def selectParentRise(self):
        """Selecție ciclică (round-robin)."""
        arg = self.parent_arg
        self.parent_arg = (self.parent_arg + 1) % self.POPULATION_SIZE
        return arg

    def selectParentMixt(self, size_subset=7, p_select=None):
        """
        Mixt între choice / wheel / tour / rise.
        p_select = distribuția probabilităților pentru cele 4 metode.
        """
        cond = np.random.choice([0, 1, 2, 3], p=p_select)

        if cond == 0:
            return self.selectParentChoice()

        elif cond == 1:
            return self.selectParentWheel()

        elif cond == 2:
            return self.selectParentTour(size_subset)

        elif cond == 3:
            return self.selectParentRise()

        # fallback
        return self.selectParentChoice()
