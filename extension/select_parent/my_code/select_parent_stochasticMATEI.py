#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class SelectParentStochastic(RootGA):
    """
    ONLY stochastic selection:
        choice, rand, wheel
    """

    def __init__(self, method=None, **kw):
        super().__init__()
        self.__configs = kw
        self.parent_arg = 0
        self.fitness_values = None
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"SelectParentStochastic(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """SelectParentStochastic:
    'choice' | 'rand' | 'wheel'
"""

    def __unpackMethod(self, method):
        table = {
            "choice": self.selectParentChoice,
            "rand":   self.selectParentRand,
            "wheel":  self.selectParentWheel
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
            f"Lipseste metoda '{self.__method}' pentru SelectParentStochastic"
        )

    def selectParentChoice(self):
        arg = np.random.choice(self.POPULATION_SIZE, p=self.fitness_values)
        return arg

    def selectParentRand(self):
        arg = np.random.randint(0, self.POPULATION_SIZE)
        return arg

    def selectParentWheel(self):
        current = 0
        pick    = np.random.uniform(0, 1)
        for arg, fv in enumerate(self.fitness_values, 0):
            current += fv
            if current > pick:
                break
        return arg
