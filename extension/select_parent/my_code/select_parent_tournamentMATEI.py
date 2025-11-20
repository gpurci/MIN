#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class SelectParentTournament(RootGA):
    """
    ONLY tournament / structured selection:
        tour, tour_choice, rise, mixt
    (no basic stochastic here except what mixt calls internally)
    """

    def __init__(self, method=None, **kw):
        super().__init__()
        self.__configs = kw
        self.parent_arg = 0
        self.fitness_values = None
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"SelectParentTournament(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """SelectParentTournament:
    'tour' | 'tour_choice' | 'rise' | 'mixt'
"""

    def __unpackMethod(self, method):
        table = {
            "tour":        self.selectParentTour,
            "tour_choice": self.selectParentTourChoice,
            "rise":        self.selectParentRise,
            "mixt":        self.selectParentMixt
        }
        return table.get(method, self.selectParentAbstract)

    def setParameters(self, **kw):
        super().setParameters(**kw)

    def startEpoch(self, fitness_values):
        total_fitness = fitness_values.sum()
        if total_fitness != 0:
            size = self.POPULATION_SIZE - int(self.SELECT_RATE * self.POPULATION_SIZE)
            args_weaks = np.argpartition(fitness_values, size)[:size]
            fitness_values[args_weaks] = 0.0
            total_fitness = fitness_values.sum()
            self.fitness_values = fitness_values / total_fitness
        else:
            self.fitness_values = np.full(
                fitness_values.shape[0],
                1.0 / self.POPULATION_SIZE,
                dtype=np.float32
            )

    def __call__(self, **kw):
        return self.__fn()

    def selectParentAbstract(self, **kw):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru SelectParentTournament"
        )

    # note: these rely on fitness_values set in startEpoch
    def selectParentTour(self, size_subset=7):
        args_k = np.random.randint(0, self.POPULATION_SIZE, size=size_subset)
        arg = np.argmax(self.fitness_values[args_k])
        return args_k[arg]

    def selectParentTourChoice(self, size_subset=7):
        args_k = np.random.choice(self.POPULATION_SIZE, size=size_subset, p=self.fitness_values)
        arg = np.argmax(self.fitness_values[args_k])
        return args_k[arg]

    def selectParentRise(self):
        arg = self.parent_arg
        self.parent_arg = (self.parent_arg + 1) % self.POPULATION_SIZE
        return arg

    def selectParentMixt(self, size_subset=7, p_select=None):
        cond = np.random.choice([0, 1, 2, 3], p=p_select)
        if cond == 0:
            return np.random.choice(self.POPULATION_SIZE, p=self.fitness_values)
        if cond == 1:
            # wheel
            current = 0
            pick = np.random.uniform(0, 1)
            for arg, fv in enumerate(self.fitness_values, 0):
                current += fv
                if current > pick:
                    break
            return arg
        if cond == 2:
            return self.selectParentTour(size_subset)
        return self.selectParentRise()
