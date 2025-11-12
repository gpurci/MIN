
    # TTP problem metrics ---------------------
    def computeSpeedTTP(self, Wcur, vmax, vmin, Wmax):
        """
        viteza curenta in functie de weight (formula TTP)
        """
        frac = min(1.0, Wcur/Wmax)
        v = vmax - frac*(vmax-vmin)
        return v if v > 1e-9 else 1e-9

    def getIndividDistanceTTP(self, individ, distance_matrix=None):
        """
        distanta rutelor TTP (daca inchizi ruta)
        use: metrics.getIndividDistanceTTP(individ)
        """
        D = distance_matrix if (distance_matrix is not None) else self.distance

        distances = D[individ[:-1], individ[1:]]
        return distances.sum() + D[individ[-1], individ[0]]
    
    def metricsTTP(self, population):
        N = population.shape[0]
        distances = np.zeros(N, dtype=np.float32)
        profits   = np.zeros(N, dtype=np.float32)
        times     = np.zeros(N, dtype=np.float32)

        ord_cities = np.arange(self.dataset["item_profit"].shape[0], dtype=np.int32)

        for i, individ in enumerate(population):

            Wcur  = 0.0
            T     = 0.0
            P     = 0.0


        pos_genoms = np.full(individ.shape, individ.shape[0]+1, dtype=np.int32)

        args_pos, args_genoms = np.nonzero(individ == ord_cities)
        pos_genoms[args_pos]  = args_genoms
        #print("pos_genoms", pos_genoms)
        diff = np.abs(pos_genoms[:-1] - pos_genoms[1:])
        print("diff", diff)
        return (diff == 1)


            for k in range(self.GENOME_LENGTH-1):
                city = individ[k]


                # adds profit
                for (city_k, weight_k, profit_k) in self.items:
                    if city_k == city:
                        P += profit_k
                        Wcur += weight_k

                v = self.computeSpeedTTP(Wcur, self.v_max, self.v_min, self.W)
                T += self.distance[individ[k], individ[k+1]] / v

            distances[i] = self.getIndividDistanceTTP(individ)
            profits[i]   = P
            times[i]     = T

        self.metrics_values = {
            "distances" : distances,
            "profits"   : profits,
            "times"     : times
        }
        return self.metrics_values
    
    def getScoreTTP(self, population, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        individ  = population[arg_best]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ

        score = self.getIndividDistanceTTP(individ, self.distance)

        return {"score": score, "best_fitness": best_fitness}
    # TTP problem finish =================================
