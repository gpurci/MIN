#!/usr/bin/python

import numpy as np
import yaml
import sys

def remove_modules(modules_name, *arg):
    if (modules_name in sys.modules):
        del sys.modules[modules_name]
        for key in arg:
            tmp_modules = "{}.{}".format(modules_name, key)
            del sys.modules[tmp_modules]

remove_modules("root_GA")
remove_modules("callback")
remove_modules("crossover")
remove_modules("fitness")
remove_modules("init_population")
remove_modules("metrics")
remove_modules("mutate")
remove_modules("select_parent")

from root_GA import *
from callback import *
from crossover import *
from fitness import *
from init_population import *
from metrics import *
from mutate import *
from select_parent import *

class GeneticAlgorithm(RootGA):
    """
    Managerul de configuratie al algoritmului genetic,
    """
    def __init__(self, name="", extern_commnad_file="", **configs):
        super().__init__()
        self.__name = name
        self.__configs = configs
        self.__setConfig(**configs)
        self.__score_evolution = np.zeros(5, dtype=np.float32)
        self.__last_mutation_rate = None
        self.__extern_commnad_file = extern_commnad_file
        self.__is_stop = False

    def __str__(self):
        str_info = "Name: {}\n{}".format(self.__name, super().__str__())
        str_info += "\nConfigs:"
        for key in self.__configs.keys():
            str_info += "\n\t{}: {}".format(key, self.__configs[key])
        str_info += "\n"
        return str_info

    def __call__(self, population):
        """
        Manager pentru a face sincronizarea dintre toate functionalele
        """
        self.__init_command()
        # initiaizarea populatiei
        if (population is None):
            population = self.initPopulation(self.POPULATION_SIZE)
        # calculate metrics
        metric_values = self.metrics(population)
        # init fitness value
        fitness_values = self.fitness(population, metric_values)
        # obtinerea pozitiei pentru elite
        args_elite = self.getArgsElite(fitness_values)

        # evolutia generatiilor
        for generation in range(self.GENERATIONS):
            # pentru oprire fortata
            if (self.__is_stop):
                break
            # nasterea unei noi generatii
            new_population = []
            # start selectie populatie
            self.selectParent1.startEpoch(fitness_values)
            self.selectParent2.startEpoch(fitness_values)
            for _ in range(self.POPULATION_SIZE):
                # selectarea positia parinte 1
                arg_parent1 = self.selectParent1()
                # selectarea positia parinte 1
                arg_parent2 = self.selectParent2()
                # obtinerea parintilor
                parent1 = population[arg_parent1]
                parent2 = population[arg_parent2]
                # incrucisarea parintilor
                offspring = self.crossover(parent1, parent2)
                # mutatii
                offspring = self.mutate(parent1, parent2, offspring) # in_place operation
                # adauga urmasii la noua generatie
                new_population.append(offspring)
            # obtinerea indivizilor ce fac parte din elita
            elite_individs = population[args_elite]
            elite_fitness  = fitness_values[args_elite]
            # schimbarea generatiei
            population = np.array(new_population)
            # calculate metrics
            metric_values  = self.metrics(population)
            # calculare fitness
            fitness_values = self.fitness(population, metric_values)
            # adaugare elita in noua populatie
            population = self.setElitesByFitness(population, fitness_values, elite_individs)
            # set elite fitness
            #fitness_values[args_elite] = elite_fitness
            # obtinerea pozitiei pentru elite
            args_elite = self.getArgsElite(fitness_values)
            # calculare metrici
            scores = self.metrics.getScore(population, fitness_values)
            #self.log(population, fitness_values, args_elite, elite_individs, best_distance)
            # adaugare stres in populatie atunci cand lipseste progresul
            self.stres(scores)
            # afisare metrici
            self.showMetrics(generation, scores)
            # salveaza istoricul
            self.callback(generation, scores)
            # evolution
            self.evolutionMonitor(scores)

        return self.metrics.getBestIndivid(), population

    def __setConfig(self, **configs):
        # configurare metrici
        config       = configs.get("metric", {""})
        method       = config.get("metric", None)
        self.metrics = Metrics(method, )
        # configurare initializare populatie
        config       = configs.get("init_population", None)
        self.initPopulation = InitPopulation(config, self.metrics)
        # configurare fitness
        config       = configs.get("fitness", None)
        self.fitness = Fitness(config)
        # configurate selectie parinti
        config       = configs.get("select_parent", {"test1":0, "test2":0})
        conf_select1 = config.get("select_parent1", None)
        self.selectParent1 = SelectParent(conf_select1)
        conf_select2 = config.get("select_parent2", None)
        self.selectParent2 = SelectParent(conf_select2)
        # configurare incrucisare
        config       = configs.get("crossover", None)
        self.crossover = Crossover(config)
        # configurare mutatie
        config       = configs.get("mutate", None)
        self.mutate  = Mutate(config)
        # configurare callback salvare, istoricul de antrenare
        filename     = configs.get("callback", None)
        self.callback = Callback(filename)

    def help(self):
        info  = "'nume': numele obiectului\n"
        info += "'extern_commnad_file': numele fisierului in care vor fi adaugate comenzile externe, (oprire fortata = stop=True)\n"
        info += "'metric': "+self.metrics.help()
        info += "'init_population': "+self.initPopulation.help()
        info += "'fitness': "+self.fitness.help()
        info += "'select_parent': {'select_parent1': 'select_parent2'}: "+self.selectParent1.help()
        info += "'crossover': "+self.crossover.help()
        info += "'mutate': "+self.mutate.help()
        info += "'callback': "+self.callback.help()
        print(info)

    def setDataset(self, dataset):
        self.metrics.setDataset(dataset)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        self.metrics.setParameters(**kw)
        self.initPopulation.setParameters(**kw)
        self.fitness.setParameters(**kw)
        self.selectParent1.setParameters(**kw)
        self.selectParent2.setParameters(**kw)
        self.crossover.setParameters(**kw)
        self.mutate.setParameters(**kw)
        self.individRepair.setParameters(**kw)

    def evolutionMonitor(self, evolutionScores):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        evolutionScores - scorul evolutiei
        """
        self.__score_evolution[:-1] = self.__score_evolution[1:]
        self.__score_evolution[-1]  = evolutionScores["score"]
        if (evolutionScores["best_fitness"] == 0):
            raise Exception("Best fitness is '0'")

    def setElites(self, population, elites):
        if (population is None):
            population = self.initPopulation(self.POPULATION_SIZE)
        fitness_values = self.fitness(population)
        args = self.getArgsWeaks(fitness_values, self.ELITE_SIZE)
        population[args] = elites
        return population

    def setElitesByFitness(self, population, fitness_values, elites):
        if (population is None):
            population = self.initPopulation(self.POPULATION_SIZE)
        args = self.getArgsWeaks(fitness_values, self.ELITE_SIZE)
        population[args] = elites
        return population

    def clcMetrics(self, population, fitness_values):
        """
        Calculare metrici:
            population - populatia compusa dintr-o lista de indivizi
            fitness_values - valorile fitnes pentru fiecare individ
        """
        raise NameError("Nu este implementata calcularea metricilor")

    def showMetrics(self, generation, d_info):
        """Afisare metrici"""
        metric_info ="Name:{}, Generatia: {}, ".format(self.__name, generation)
        for key in d_info.keys():
            val = d_info[key]
            if (isinstance(val, float)):
                val = round(val, 3)
            metric_info +="{}: {}, ".format(key, val)
        print(metric_info)

    def stres(self, evolutionScores):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        evolutionScores - scorul evolutiei
        """
        check_distance = np.allclose(self.__score_evolution.mean(), evolutionScores["score"], rtol=1e-03, atol=1e-08)
        #print("distance evolution {}, distance {}".format(check_distance, best_distance))
        if (check_distance):
            self.__score_evolution[:] = 0
            self.__last_mutation_rate = self.MUTATION_RATE
            print("evolutionScores {}".format(evolutionScores))
            self.setParameters(MUTATION_RATE=1.)
            self.mutate.increaseSubsetSize()
            self.externCommand()
        else:
            self.mutate.decreaseSubsetSize()
            if (self.__last_mutation_rate is not None):
                self.setParameters(MUTATION_RATE=self.__last_mutation_rate)

    def getArgsWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        args = np.argpartition(fitness_values, size)
        return args[:size]

    def getArgsElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        args = np.argpartition(fitness_values,-self.ELITE_SIZE)
        args = args[-self.ELITE_SIZE:]
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
