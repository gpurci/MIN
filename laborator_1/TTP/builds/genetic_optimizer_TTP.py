#!/usr/bin/python

import numpy as np
from builds.ttp_generator import TTPGenerator

class GeneticAOTSP(object):
    # constante pentru setarea algoritmului
    POPULATION_SIZE = 100 # numarul populatiei
    GENOME_LENGTH   = 4 # numarul de orase
    MUTATION_RATE   = 0.01  # threshold-ul pentru a face o mutatie genetica
    CROSSOVER_RATE  = 0.5   # threshold-ul pentru incrucisarea parintilor
    SELECT_RATE     = 0.8   # threshold-ul de selectie, selectare dupa compatibilitate sau dupa probabilitate
    K_DISTANCE      = 0.1   # coeficientul de pondere a distantei si inhibare a numarului de orase
    K_NBR_CITY      = 0.1   # coeficientul de pondere a distantei si inhibare a numarului de orase
    K_BEST          = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mare scor
    K_WRONG         = 5     # salveaza pentru urmatoarea generatie numarul de indivizi, cu cel mai mic scor
    GENERATIONS     = 500 # numarul de generatii

    def __init__(self, max_cities=-1):
        self.max_cities = max_cities

    def __call__(self, distance: "np.array", nodes_csv_path: str, items_csv_path: str, population=None, start_city=-1):
            # save map
            self.__run_checks(distance)
            # initiaizarea populatiei
            if (population is None):
                population = self.initPopulation(nodes_csv_path, items_csv_path,
                                            population_target=GeneticAOTSP.POPULATION_SIZE)
            self.__run_inits(population)
            # init fitness value
            fitness_values = self.fitness(population)
            # obtinerea pozitiei pentru indivizii extrimali
            arg_extreme = self.getKExtreme(fitness_values)

            # evolutia generatiilor
            for generation in range(GeneticAOTSP.GENERATIONS):
                # nasterea unei noi generatii
                new_population = []
                # selectarea unui parinte
                size_parents = GeneticAOTSP.POPULATION_SIZE-arg_extreme.shape[0]
                arg_parents1 = self.selectParent1(population, fitness_values, size_parents//3)
                valid_arg_parents2 = self.selectValidPopulation(arg_parents1, fitness_values[arg_parents1])
                valid_population_parents2 = population[valid_arg_parents2]
                print("valid_population_parents2 {}".format(valid_population_parents2.shape))
                valid_fitness_parents2    = fitness_values[valid_arg_parents2]
                for arg_parent1 in arg_parents1:
                    # selectarea celui de al doilea parinte
                    childrens = []
                    for _ in range(3):
                        arg_parent2 = self.selectParent2(valid_population_parents2, valid_fitness_parents2, fitness_values[arg_parent1])
                        parent1 = population[arg_parent1]
                        parent2 = valid_population_parents2[arg_parent2]
                        print("parent1 {}, parent2 {}, arg_parent2 {}".format(parent1, parent2, arg_parent2))
                        # incrucisarea parintilor
                        for __ in range(3):
                            child = self.crossover(parent1, parent2)
                            # mutatii
                            child = self.mutate(child)
                            childrens.append(child)
                    childrens = np.array(childrens, dtype=np.int32)
                    childs_fitness_values = self.fitness(childrens)
                    print("childrens {}, childs_fitness_values {}, shape {}".format(childrens, childs_fitness_values, childrens.shape))
                    arg_childrens = self.getKBest(childs_fitness_values, 3)
                    print("childrens {}, shape {}".format(childrens, childrens.shape))
                    new_population.append(childrens[arg_childrens])
                # salvarea indivizilor extrimali
                extreme_population  = population[arg_extreme]
                # schimbarea generatiei
                #population = np.array(new_population, dtype=np.int32)
                new_population.append(extreme_population)
                # integrarea indivizilor extrimali in noua generatie
                population = np.concatenate(new_population, axis=0)
                # update la valorile fitness
                fitness_values = self.fitness(population)
                # obtinerea pozitiilor pentru indivizii extrimali din noua generatie
                arg_extreme = self.getKExtreme(fitness_values)
                # obtinerea celui mai bun individ
                arg_best = self.getBestRoute(fitness_values)
                # selectarea celei mai bune rute
                best_route = population[arg_best]
                distance = self.getIndividDistance(best_route)
                number_city = self.getIndividNumberCities(best_route)
                # selectarea celei mai mici distante din intreaga populatie
                best_distance = self.getBestDistance(population)
                # selectarea celui mai mare numar de orase din intreaga populatie
                best_number_city = self.getBestNumberCities(population)

                # prezinta metricile
                metric_info ="""Generatia: {}, Distanta: {:.3f}, Numarul oraselor {}, Best Distanta: {:.3f}, Best Numarul oraselor {}, Min distance {:.3f}""".format(
                    generation, distance, number_city, best_distance, best_number_city, self.__min_distance)
                print(metric_info)
            return best_route, population

    def __run_checks(self, distance):
        if (distance is not None):
            self.distance = distance
            #self.total_distance = np.sum(self.distance, axis=None)
            # update numarul de orase
            GeneticAOTSP.GENOME_LENGTH = self.distance.shape[0]
        else:
            raise NameError("Lipseste 'distance' este un numar null")
        if (self.max_cities < 0):
            self.max_cities = 0

    def __run_inits(self, population):
        # calculeaza distanta
        distances = self.getDistances(population)
        self.__min_distance = distances.min()

    # viteza TTP standard (liniara în greutatea curenta)
    @staticmethod
    def _speed_linear(vmax: float, vmin: float, Wmax: float, Wcur: float) -> float:
        if Wmax <= 0: return vmax
        frac = max(0.0, min(1.0, Wcur / Wmax))
        return vmax - frac * (vmax - vmin)

    # Initializare BEAM pentru populatie: scor = profit(next) - λ * travel_time(cur->next)
    # fallback random daca nu atingem targetul
    def initPopulationTooComplexToRun( self,
                        nodes_csv_path: str,
                        items_csv_path: str,
                        *,

                        vmax: float = 1.0,
                        vmin: float = 0.1,
                        Wmax: float = 25936.0,

                        population_target: int | None = None,  # daca None -> 4 * n
                        beam_width: int = 2,
                        lambda_time: float = 0.1,
                        start_city: int | None = None,
                        seed: int | None = None,
                        allow_pick_in_start_city: bool = False  # nu se stie daca se poate sau nu
                        ) -> np.ndarray:
        """
        Genereaza o populatie initiala “inteligenta” cu beam-search:
          - scor(next) = profit[next] - lambda_time * (distance[cur,next] / speed_cur)
          - speed_cur depinde de greutatea acumulata deja (standard TTP: liniar între vmax și vmin)
          - in orasul de START nu se fura. Prima decizie de picking este in orasul al doilea vizitat.
            (putem permite explicit cu allow_pick_in_start_city=True daca dorim)

        Returneaza:
          - population: np.ndarray shape (population_target, GENOME_LENGTH) cu ultimul oras egal cu primul (ciclu inchis)
        """
        if seed is not None:
            np.random.seed(seed)
            

        # Citeste CSV-urile si pregateste distantele + (profit, weight) pe oras
        loader = TTPGenerator(0, 0)
        loader.load_ttp_csv(nodes_csv_path, items_csv_path)

        # copy into GA instance
        self.coords       = loader.coords
        self.distance     = loader.distance
        self.item_profit  = loader.item_profit
        self.item_weight  = loader.item_weight
        GeneticAOTSP.GENOME_LENGTH = self.coords.shape[0]
        n = GeneticAOTSP.GENOME_LENGTH

        # Dimensionare implicita populatie -> poate fi si 2*n
        if population_target is None:
            population_target = 2 * n

        population: list[np.ndarray] = []
        seen = set()  # pentru a evita rute duplicate (tuple de orase)

        # Alegem orase de start (1 sau mai multe pentru diversitate)
        if start_city is not None:
            start_candidates = [start_city]
        else:
            # ia 20 start cities random pt diversitate
            K_starts = 1
            start_candidates = np.random.choice(n, size=min(K_starts, n), replace=False)
        # Beam-search pe fiecare start pana umplem populatia
        for s in start_candidates:
            if len(population) >= population_target:
                break

            # beam contine tupluri: (path:list[int], current_weight:float, total_profit:float)
            beam = [([s], 0.0, 0.0)]

            # in start city nu furam -> nu sunt sigur daca se fura automat din primul oras
            can_pick_in_city0 = bool(allow_pick_in_start_city)

            while beam and len(population) < population_target:
                next_beam = []

                for path, Wcur, Ptot in beam:
                    cur = path[-1]

                    # marcam orasele vizitate
                    visited = np.zeros(n, dtype=bool)
                    visited[path] = True

                    # daca am vizitat toate orasele, inchidem turul si salvam individul
                    if visited.all():
                        full_path = path + [path[0]]  # ultimul = primul
                        key = tuple(full_path)
                        if key not in seen:
                            seen.add(key)
                            population.append(np.array(full_path, dtype=np.int32))
                            if len(population) >= population_target:
                                break
                        continue

                    # candidati = orase nevizitate
                    cand = np.where(~visited)[0]
                    if cand.size == 0:
                        continue

                    # --- LIMITAREA COMPLEXITATII ---
                    # doar cei mai apropiati 2 orase nevizitate (in loc de toate 279)
                    K = 2
                    d = self.distance[cur, cand]
                    order = np.argsort(d)
                    cand = cand[ order[:K] ]

                    # viteza curenta (inainte de a te deplasa catre urmatorul oras)
                    v_cur = GeneticAOTSP._speed_linear(vmax, vmin, Wmax, Wcur)
                    v_cur = max(v_cur, 1e-12)  # protectie numerica

                    # calculează scor pentru fiecare candidat
                    scored = []
                    for j in cand:
                        # timp de mers cur->j
                        t = self.distance[cur, j] / v_cur

                        # profit/greutate posibile în orasul j (un singur item per city)
                        p_j = self.item_profit[j]
                        w_j = self.item_weight[j]

                        # In orasul de START NU furam.
                        # -> asta afectează doar prima decizie, cand path are lungimea 1.
                        can_take_here = True
                        if len(path) == 1 and not can_pick_in_city0:
                            # Suntem pe primul pas: NU luam item din primul oras.
                            pass

                        # Decizia locala: daca itemul din j incape, scorul include profitul; altfel, doar timpul penalizeaza.
                        will_fit = (w_j > 0.0) and ((Wcur + w_j) <= Wmax)
                        # scor = profit (daca putem lua) - λ * timp
                        score = (p_j if (will_fit and can_take_here) else 0.0) - (lambda_time * t)
                        scored.append((score, j, p_j, w_j, t))

                    # alegem top-k după scor
                    scored.sort(key=lambda tup: tup[0], reverse=True)
                    for (score, j, p_j, w_j, t) in scored[:beam_width]:
                        # Mutam: cur -> j
                        # La sosire in j, putem decide sa luam item daca incape.
                        take = (p_j > 0.0) and ((Wcur + w_j) <= Wmax)
                        new_W = Wcur + (w_j if take else 0.0)
                        new_P = Ptot + (p_j if take else 0.0)

                        next_beam.append((path + [j], new_W, new_P))

                beam = next_beam

        # Daca beam nu a produs suficient — completam cu permutari random (safety net)
        if len(population) < population_target:
            deficit = population_target - len(population)
            population.extend(self._fallback_random_population(deficit))

        # Returnam exact population_target indivizi, ultimul == primul
        return np.stack(population[:population_target], axis=0)

    def initPopulation(self,
                    nodes_csv_path,
                    items_csv_path,
                    *,
                    population_target=2,
                    seed=None):

        if seed is not None:
            np.random.seed(seed)

        # load
        loader = TTPGenerator(0,0)
        loader.load_ttp_csv(nodes_csv_path, items_csv_path)
        self.coords = loader.coords
        self.distance = loader.distance
        self.item_profit = loader.item_profit
        self.item_weight = loader.item_weight
        n = loader.coords.shape[0]
        GeneticAOTSP.GENOME_LENGTH = n

        population = []

        starts = np.random.choice(n, size=population_target, replace=False)

        for s in starts:
            visited = np.zeros(n, dtype=bool)
            path = [s]
            visited[s] = True
            cur = s

            # greedy nearest neighbour full tour
            for _ in range(n-1):
                cand = np.where(~visited)[0]
                j = cand[np.argmin(self.distance[cur, cand])]
                path.append(j)
                visited[j] = True
                cur = j

            # close tour
            path.append(path[0])

            population.append(path)

        return np.array(population, dtype=np.int32)
    # Completeaza cu rute random (ultimul = primul), daca beam nu atinge targetul
    def _fallback_random_population(self, k: int) -> list[np.ndarray]:
        n = GeneticAOTSP.GENOME_LENGTH
        out = []
        for _ in range(k):
            perm = np.random.permutation(n)
            perm = np.concatenate([perm, [perm[0]]]).astype(np.int32)
            out.append(perm)
        return out

    def selectValidPopulation(self, arg_parents1, fitness_values_parents1):
        # select valid parents for parents2, from list of valid parents1
        # 1/3 from parents1 is valid as a parents2
        # create mask of valid parents2 from valid parents1
        fitness_values_parents1 = fitness_values_parents1/fitness_values_parents1.sum()
        arg_valid_id_parents1   = np.random.choice(arg_parents1.shape[0], size=arg_parents1.shape[0]//3, p=fitness_values_parents1)
        # exclude valid parents1 from invalid parents selection
        mask = np.ones(arg_parents1.shape[0], dtype=bool)
        print("mask", mask.shape)
        mask[arg_valid_id_parents1] = False
        # do invalid parents1 for valid parents2
        arg_invalid_parents1 = arg_parents1[mask]
        arg_populations = np.ones(GeneticAOTSP.POPULATION_SIZE, dtype=bool)
        print("arg_populations", arg_populations.shape)
        arg_populations[arg_invalid_parents1] = False
        return np.argwhere(arg_populations).reshape(-1)

    def selectParent1(self, population, fitness_values, size_parents):
        """selectarea unui parinte aleator din populatie, bazandune pe distributia fitness valorilor"""
        # select random parent
        prob_fitness = fitness_values / fitness_values.sum()
        #print("selectie aleatorie", prob_fitness)
        args = np.random.choice(GeneticAOTSP.POPULATION_SIZE, size=size_parents, p=prob_fitness)
        return args

    def selectParent2(self, population, fitness_values, fitness_partener):
        """selectarea unui parinte aleator din populatie"""
        select_rate = np.random.uniform(low=0, high=GeneticAOTSP.SELECT_RATE, size=None)
        if (select_rate < GeneticAOTSP.SELECT_RATE): # selectie dupa compatibilitate
            total_fitness = np.sum(fitness_values, axis=None)
            pick = np.random.uniform(low=fitness_partener, high=total_fitness, size=None)
            current = 0
            for arg, fitness_value in enumerate(fitness_values, 0):
                current += fitness_value
                if (current > pick):
                    break
        else: # selectie aleatorie
            prob_fitness = fitness_values / fitness_values.sum()
            #print("selectie aleatorie", prob_fitness)
            # selecteaza argumentul parintelui 2
            arg = np.random.choice(GeneticAOTSP.POPULATION_SIZE, size=None, p=prob_fitness)
        return arg

    def crossover(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        """
        # creare un copil fara mostenire
        child = np.zeros(parent1.shape[0])
        # selectarea diapazonului de mostenire
        low   = 1
        hight = parent1.shape[0]-1
        start = np.random.randint(low=low, high=hight, size=None)
        end   = np.random.randint(low=low, high=hight, size=None)
        if (start > end):
            start, end = end, start
        # copierea rutei din primul parinte
        child[start:end] = parent1[start:end]
        # copierea rutei din cel de al doilea parinte
        child[:start] = parent2[:start]
        child[end:]   = parent2[end:]
        return child

    def mutate(self, individ):
        """Mutatia genetica a rutei"""
        # selectarea genomurile care vor fi mutate cu locul
        cond = np.random.uniform(low=0, high=1, size=None)
        if (cond < (GeneticAOTSP.MUTATION_RATE*0.3)):
            low   = 1
            hight = individ.shape[0]-1
            index1 = np.random.randint(low=low, high=hight, size=None)
            index2 = np.random.randint(low=low, high=hight, size=None)
            individ[index1], individ[index2] = individ[index2], individ[index1]
        elif (cond < (GeneticAOTSP.MUTATION_RATE)):
            low   = 1
            hight = individ.shape[0]-1
            index = np.random.randint(low=low, high=hight, size=None)
            gene  = np.random.choice(np.arange(low, hight), size=None)
            individ[index] = gene

        return individ

    def getIndividDistance(self, individ):
        """Calculul distantei rutelor"""
        #print("individ", individ)
        distances = self.distance[individ[:-1], individ[1:]]
        distance = distances.sum() + self.distance[individ[-1], individ[0]]
        return distance

    def getIndividNumberCities(self, individ):
        return np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def neighborFreq(self, individ, gene):
        arg_gene = np.argwhere(individ==gene).reshape(-1)
        if (arg_gene.shape[0]):
            freq = 1
        else:
            diff_gene = np.abs(arg_gene[:-1]-arg_gene[1:])
            nbr_gene  = diff_gene.shape[0]
            neighbor_genes = (diff_gene==1).sum()
            freq = neighbor_genes/nbr_gene
        return freq

    def neighborsFreq(self, individ):
        genes = np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None)
        freq = 0
        for gene in genes:
            freq += self.neighborFreq(individ, gene)
        return freq/individ.shape[0]

    def getBestRoute(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestDistance(self, population):
        # calculeaza distanta
        distances = self.getDistances(population)
        return distances.min()

    def getBestNumberCities(self, population):
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        return number_city.max()

    def getKBest(self, fitness_values, size_best):
        # returneaza pozitiile cu cele mai mari valori
        arg_ = np.argpartition(fitness_values,-size_best)
        return arg_[-size_best:]

    def getKExtreme(self, fitness_values):
        # returneaza pozitiile cu cele mai mari valori
        # select size of right population
        low=GeneticAOTSP.K_BEST-(GeneticAOTSP.K_BEST//3)
        k_best = np.random.randint(low=low, high=GeneticAOTSP.K_BEST, size=None)
        # select size of left population
        low=GeneticAOTSP.K_WRONG-(GeneticAOTSP.K_WRONG//3)
        k_wrong = np.random.randint(low=low, high=GeneticAOTSP.K_WRONG, size=None)
        # select size of normal population
        k_normal = k_best+k_wrong
        # select best individs
        k = fitness_values.shape[0]//2-k_normal//2
        arg_ = np.argpartition(fitness_values,-k)
        arg_normal = arg_[-k:k_normal-k]
        arg_best   = arg_[-k_best:]
        # select left individs
        arg_wrong = np.argpartition(fitness_values, k_wrong)[:k_wrong]
        return np.concatenate((arg_best, arg_normal, arg_wrong), axis=0)

    def getDistances(self, population):
        """calcularea distantei pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.getIndividDistance,
                                        axis=1,
                                        arr=population)

    def getNumberCities(self, population):
        # calculeaza numarul de orase unice
        return np.apply_along_axis(self.getIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def getNeighborFreq(self, population):
        # calculeaza numarul de orase unice
        return np.apply_along_axis(self.neighborsFreq,
                                        axis=1,
                                        arr=population)


    def fitness(self, population):
        """ valoarea la fitness este inversul distantei dintre orase si un numar cat mai mare de orase vizitate
        - se calculeaza distanta maxima pentru fiecare individ
        - se calculeaza numarul de orase unice
        """
        # calculeaza distanta
        distances = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # calculeaza frecventa vecinilor unici apropiati
        neighbor_city = self.getNeighborFreq(population)
        #print("neighbor_city", neighbor_city)
        # calculeaza coeficientul
        distances = self.__distance_norm(distances)
        number_city = self.__city_norm(number_city, GeneticAOTSP.GENOME_LENGTH)
        #print("distances: min {:.3f}, max {:.3f}, mean {:.3f}, std {:.3f}, quatile_25 {:.3f}, quatile_50 {:.3f}, quatile_75 {:.3f}".format(distances.min(), distances.max(), np.mean(distances), np.std(distances),
        #                                                          np.quantile(distances, 0.25), np.quantile(distances, 0.5), np.quantile(distances, 0.75)))
        #print("number_city", number_city)
        #print("neighbor_city", neighbor_city)
        fitness_values = GeneticAOTSP.K_DISTANCE*distances + GeneticAOTSP.K_NBR_CITY*number_city*neighbor_city
        return fitness_values

    def setParameters(self, **kw):
        GeneticAOTSP.POPULATION_SIZE = kw.get("POPULATION_SIZE", GeneticAOTSP.POPULATION_SIZE)
        GeneticAOTSP.MUTATION_RATE   = kw.get("MUTATION_RATE", GeneticAOTSP.MUTATION_RATE)
        GeneticAOTSP.CROSSOVER_RATE  = kw.get("CROSSOVER_RATE", GeneticAOTSP.CROSSOVER_RATE)
        GeneticAOTSP.SELECT_RATE     = kw.get("SELECT_RATE", GeneticAOTSP.SELECT_RATE)
        GeneticAOTSP.GENERATIONS     = kw.get("GENERATIONS", GeneticAOTSP.GENERATIONS)
        GeneticAOTSP.K_DISTANCE      = kw.get("K_DISTANCE", GeneticAOTSP.K_DISTANCE)
        GeneticAOTSP.K_NBR_CITY      = kw.get("K_NBR_CITY", GeneticAOTSP.K_NBR_CITY)
        GeneticAOTSP.K_BEST          = kw.get("K_BEST", GeneticAOTSP.K_BEST)
        GeneticAOTSP.K_WRONG         = kw.get("K_WRONG", GeneticAOTSP.K_WRONG)

    def __distance_norm(self, distances):
        #max_distance = distances.max()
        self.__min_distance = self.__min_distance*0.99 + distances.min()*0.01
        return (2*self.__min_distance)/(distances + self.__min_distance)

    def __city_norm(self, number_city, size):
        return (number_city-1)/size
