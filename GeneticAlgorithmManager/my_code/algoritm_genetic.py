#!/usr/bin/python

import numpy as np
import yaml
import warnings

from sys_function import sys_remove_modules

sys_remove_modules("root_GA")
sys_remove_modules("genoms")
sys_remove_modules("callback")
sys_remove_modules("crossover")
sys_remove_modules("fitness")
sys_remove_modules("init_population")
sys_remove_modules("metrics")
sys_remove_modules("mutate")
sys_remove_modules("select_parent")
sys_remove_modules("stres")

from root_GA import *
from genoms import *
from callback import *
from crossover import *
from fitness import *
from init_population import *
from metrics import *
from mutate import *
from select_parent import *
from stres import *

class GeneticAlgorithm(RootGA):
    """
    Managerul de configuratie al algoritmului genetic,
    """
    def __init__(self, name="", extern_commnad_file="", **configs):
        super().__init__()
        self.__name = name
        self.__FREQ_CHECK_EXTERN = 1
        self.__freq_check_extern = 0
        self.__extern_commnad_file = extern_commnad_file
        self.__is_stop = False
        self.unpackConfig(**configs)
        self.initByConfig()

    def __str__(self):
        str_info = "Name: {}\n{}".format(self.__name, super().__str__())
        str_info += "\nExtern comand filename: '{}'".format(self.__extern_commnad_file)
        str_info += "\nConfigs: {}".format(self.__configs)
        for function in self.__functions:
            str_info += "\n{}".format(str(function))
        str_info += "\n"
        return str_info

    def __call__(self):
        """
        Manager pentru a face sincronizarea dintre toate functionalele
        """
        self.__init_command()
        # initiaizarea populatiei
        if (self.__genoms.isGenoms()==False):
            self.initPopulation(self.POPULATION_SIZE, self.__genoms)
        # calculate metrics
        metric_values  = self.metrics(self.__genoms)
        # init fitness value
        fitness_values = self.fitness(metric_values)
        # obtinerea pozitiei pentru elite
        args_elite = self.getArgsElite(fitness_values)
        # evolutia generatiilor
        for generation in range(self.GENERATIONS):
            # pentru oprire fortata
            if (self.__is_stop):
                break
            # start selectie populatie
            self.selectParent1.startEpoch(fitness_values)
            self.selectParent2.startEpoch(fitness_values)
            for _ in range(self.POPULATION_SIZE-self.ELITE_SIZE):
                # selectarea positia parinte 1
                arg_parent1 = self.selectParent1()
                # selectarea positia parinte 2
                arg_parent2 = self.selectParent2()
                # obtinerea parintilor
                parent1 = self.__genoms[arg_parent1]
                parent2 = self.__genoms[arg_parent2]
                # incrucisarea parintilor
                offspring = self.crossover(parent1, parent2)
                # mutatii
                offspring = self.mutate(parent1, parent2, offspring) # in_place operation
                # adauga urmasii la noua generatie
                self.__genoms.append(offspring)
            else:
                for arg_elite in args_elite:
                    elite_individ = self.__genoms[arg_elite]
                    # adauga elita la noua generatie
                    self.__genoms.append(elite_individ)
                else:
                    args = np.arange(self.POPULATION_SIZE-self.ELITE_SIZE, self.POPULATION_SIZE, dtype=np.int32)
                    self.__genoms.setElitePos(args)

            # schimbarea generatiei
            self.__genoms.save()
            # calculate metrics
            metric_values  = self.metrics(self.__genoms)
            # calculare fitness
            fitness_values = self.fitness(metric_values)
            # obtinerea pozitiei pentru elite
            args_elite = self.getArgsElite(fitness_values)
            # calculare metrici
            scores = self.metrics.getScore(self.__genoms, fitness_values)
            # adaugare stres in populatie atunci cand lipseste progresul
            self.stres(self.__genoms, scores)
            # afisare metrici
            self.showMetrics(generation, scores)
            # salveaza istoricul
            self.callback(generation, scores)
            # evolution
            self.evolutionMonitor(scores)

        return self.__genoms.getBest()

    def population(self):
        return self.__genoms.population()

    def setPopulation(self, population):
        self.__genoms.setPopulation(population)

    def __unpackConfigure(self, str_functia, **configs):
        method, method_configs = None, {}
        if (configs is not None):
            method_configs = configs.get(str_functia, None)
            if (method_configs is not None):
                method = method_configs.pop("method", None)
            else:
                method_configs = {}
                warnings.warn("\n\nLipseste metoda, pentru functia: '{}'".format(str_functia))
        else:
            warnings.warn("\n\nLipseste configs: 'GeneticAlgorithm'")

        return method, method_configs

    def initByConfig(self):
        if (self.__configs is not None):
            #
            self.__FREQ_CHECK_EXTERN = self.__configs.get("freq_check_extern", 5)
            #
            self.__update_elite_fitness = self.__configs.get("update_elite_fitness", True)
        else:
            self.__FREQ_CHECK_EXTERN = 5
            #
            self.__update_elite_fitness = True

    def unpackConfig(self, **configs):
        # salveaza configuratiile
        self.__functions = []
        # configureaza genoms
        config = configs.get("genoms", {})
        self.__genoms = Genoms(genome_lenght=self.GENOME_LENGTH, **config)
        self.__functions.append(self.__genoms)
        # configurare metrici
        extern_fn = configs.get("metric", None)
        self.metrics = Metrics(extern_fn)
        self.__functions.append(self.metrics)
        # configurare initializare populatie
        extern_fn = configs.get("init_population", None)
        self.initPopulation = InitPopulation(extern_fn)
        self.__functions.append(self.initPopulation)
        # configurare stres
        extern_fn  = configs.get("stres", None)
        self.stres = Stres(extern_fn)
        self.__functions.append(self.stres)
        # configurare fitness
        extern_fn = configs.get("fitness", None)
        self.fitness = Fitness(extern_fn)
        self.__functions.append(self.fitness)
        # configurate selectie parinti
        extern_fn = configs.get("select_parent1", None)
        self.selectParent1 = SelectParent(extern_fn, name="SelectParent1")
        self.__functions.append(self.selectParent1)
        extern_fn = configs.get("select_parent2", None)
        self.selectParent2 = SelectParent(extern_fn, name="SelectParent2")
        self.__functions.append(self.selectParent2)
        # configurare incrucisare
        method, method_configs = self.__unpackConfigure("crossover", **configs)
        self.crossover = Crossover(self.__genoms, method, **method_configs)
        self.__functions.append(self.crossover)
        # configurare mutatie
        method, method_configs = self.__unpackConfigure("mutate", **configs)
        self.mutate   = Mutate(self.__genoms, method, **method_configs)
        self.__functions.append(self.mutate)
        # configurare callback salvare, istoricul de antrenare
        config        = configs.get("callback", {})
        self.callback = Callback(**config)
        self.__functions.append(self.callback)
        # configurare manager salvare, istoricul de antrenare
        self.__configs = configs.get("manager", None)

    def help(self):
        info  = "'nume': numele obiectului\n"
        info += "'extern_commnad_file': numele fisierului in care vor fi adaugate comenzile externe, (oprire fortata = stop=True)\n"
        info += "'manager': {\"freq_check_extern\":5, \"update_elite_fitness\": True}\n"
        info += "'genoms': "+self.__genoms.help()
        info += "'metric': "+self.metrics.help()
        info += "'init_population': "+self.initPopulation.help()
        info += "'fitness': "+self.fitness.help()
        info += "'select_parent': {'select_parent1': 'select_parent2'}: "+self.selectParent1.help()
        info += "'crossover': "+self.crossover.help()
        info += "'mutate': "+self.mutate.help()
        info += "'stres': "+self.stres.help()
        info += "'callback': "+self.callback.help()
        print(info)

    def setParameters(self, **kw):
        print("setParameters: {}".format(kw))
        # 
        GENOME_LENGTH   = kw.get("GENOME_LENGTH", None)
        POPULATION_SIZE = kw.get("POPULATION_SIZE", None)
        if (GENOME_LENGTH is not None):
            self.__genoms.setGenomeLenght(GENOME_LENGTH)
        if (POPULATION_SIZE is not None):
            if (self.POPULATION_SIZE > POPULATION_SIZE):
                # calculate metrics
                metric_values  = self.metrics(self.__genoms)
                # init fitness value
                fitness_values = self.fitness(metric_values)
                args = self.getArgsBest(fitness_values, POPULATION_SIZE)
                population = self.__genoms[args]
                self.__genoms.setPopulation(population)
                print("update population {}".format(self.__genoms.shape))
        # 
        super().setParameters(**kw)
        self.metrics.setParameters(**kw)
        self.initPopulation.setParameters(**kw)
        self.fitness.setParameters(**kw)
        self.selectParent1.setParameters(**kw)
        self.selectParent2.setParameters(**kw)
        self.crossover.setParameters(**kw)
        self.mutate.setParameters(**kw)
        self.stres.setParameters(**kw)

    def evolutionMonitor(self, evolution_scores):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        evolution_scores - scorul evolutiei
        """
        if (self.__freq_check_extern < self.__FREQ_CHECK_EXTERN):
            self.__freq_check_extern = 0
            self.externCommand()
        else:
            self.__freq_check_extern += 1
        if (evolution_scores["best_fitness"] < 0):
            raise Exception("Best fitness is less than '0'")

    # Fitness.__call__ always requires BOTH population AND metric_values.
    # => we must compute metrics first, then compute fitness again.
    def setElites(self, elites):
        if (self.__genoms.isGenoms()==False):
            self.initPopulation(self.POPULATION_SIZE, self.__genoms)

        if (elites.shape[0] > 0):
            # MUST compute metrics first
            metric_values  = self.metrics(self.__genoms)
            fitness_values = self.fitness(metric_values)
            # set elites
            args = self.getArgsWeaks(fitness_values, elites.shape[0])
            self.__genoms[args] = elites
            self.__genoms.setElitePos(args)

    def setElitesByFitness(self, fitness_values, elites, fitness_elites=None):
        if (self.__genoms.isGenoms()==False):
            self.initPopulation(self.POPULATION_SIZE, self.__genoms)

        # here fitness_values already exists (caller passed it)
        if (elites.shape[0] > 0):
            args = self.getArgsWeaks(fitness_values, elites.shape[0])
            self.__genoms[args] = elites
            self.__genoms.setElitePos(args)
            if (fitness_elites is not None):
                fitness_values[args] = fitness_elites

    def showMetrics(self, generation, d_info):
        """Afisare metrici"""
        metric_info ="Name:{}, Generatia: {}, ".format(self.__name, generation)
        for key in d_info.keys():
            val = d_info[key]
            if (isinstance(val, float)):
                val = round(val, 3)
            metric_info +="{}: {}, ".format(key, val)
        print(metric_info)

    def getArgsWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        if (size > 0):
            args = np.argpartition(fitness_values, size)[:size]
        else:
            args = np.array([], dtype=np.int32)
        return args

    def getArgsBest(self, fitness_values, size):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        if (size > 0):
            args = np.argpartition(fitness_values, -size)
            args = args[-size:]
        else:
            args = np.array([], dtype=np.int32)
        return args

    def getArgsElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        if (self.ELITE_SIZE > 0):
            args = np.argpartition(fitness_values, -self.ELITE_SIZE)
            args = args[-self.ELITE_SIZE:]
        else:
            args = np.array([], dtype=np.int32)
        return args

    def externCommand(self):
        command_dict = self.__read_command_yaml_file()
        self.__is_stop = command_dict["stop"]

    def __read_command_yaml_file(self):
        if (isinstance(self.__extern_commnad_file, str) and (Path(self.__extern_commnad_file).is_file())):
            with open(self.__extern_commnad_file) as file :
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                command_dict = yaml.load(file, Loader=yaml.FullLoader)
        else :
            command_dict = {"stop":False}
        return command_dict

    def __init_command(self) :
        if (isinstance(self.__extern_commnad_file, str) and (not Path(self.__extern_commnad_file).is_file())):
            # create a config file
            Path(self.__extern_commnad_file).touch(mode=0o666, exist_ok=True)
        if (isinstance(self.__extern_commnad_file, str) and (Path(self.__extern_commnad_file).is_file())):
            # save default rating in yaml file
            commands      = "stop : {}".format(False)
            yaml_commands = yaml.safe_load(commands)

            with open(self.__extern_commnad_file, "w") as file :
                yaml.dump(yaml_commands, file)
