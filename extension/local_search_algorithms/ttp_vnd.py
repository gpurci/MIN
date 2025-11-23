#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA

from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.three_opt import ThreeOpt
from extension.local_search_algorithms.kp_greedy import TTPKPLocalSearch


class TTPVNDLocalSearch(RootGA):
    """
    FULL TTP-aware VND:
       - Improves TSP route using (2-opt, or-opt, 3-opt)
       - Improves KP bitstring using greedy repair (TTPKPLocalSearch)
       - Evaluates REAL TTP objective:
            profit - alpha * travel_time(weight-dependent)

    Offspring format (GA style):
       __call__(p1, p2, offspring, **cfg)
       offspring = {"tsp": np.array(n), "kp": np.array(n)}
    """

    def __init__(
        self,
        dataset=None,
        use_kp_ls=True,
        v_max=1.0,
        v_min=0.1,
        W=25936,
        alpha=0.01,
        max_rounds=3,
        **configs
    ):
        super().__init__()

        if dataset is None:
            raise ValueError("TTPVNDLocalSearch requires dataset")

        self.dataset = dataset
        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        self.v_max = v_max
        self.v_min = v_min
        self.Wmax  = W
        self.alpha = alpha
        self.max_rounds = max_rounds
        self.use_kp_ls  = use_kp_ls
        self.__configs  = configs

        # route operators (GA-style: op(p1, p2, route))
        self.ops = [
            TwoOpt("two_opt_LS", dataset),
            OrOpt("or_opt_restrict", dataset),
            ThreeOpt("three_opt_restrict", dataset),
        ]

        # genome-level KP improver: takes {"tsp","kp"} and returns {"tsp","kp"}
        self.kp_ls = TTPKPLocalSearch(dataset)

    def __str__(self):
        return f"TTPVNDLocalSearch(max_rounds={self.max_rounds}, alpha={self.alpha})"

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for op in self.ops:
            if hasattr(op, "setParameters"):
                op.setParameters(**kw)
        if hasattr(self.kp_ls, "setParameters"):
            self.kp_ls.setParameters(**kw)

    # -----------------------------------------------------------
    #   TTP scoring: profit - alpha * (distance / speed)
    # -----------------------------------------------------------
    def _compute_ttp_score(self, route, kp):
        d = self.distance
        p = self.item_profit
        w = self.item_weight

        v_max, v_min = self.v_max, self.v_min
        Wmax  = self.Wmax
        alpha = self.alpha

        Wcur   = 0.0
        time   = 0.0
        profit = 0.0
        n = len(route)

        for idx in range(n):
            city = route[idx]
            nxt  = route[(idx + 1) % n]

            # pick item
            if kp[city] == 1:
                profit += p[city]
                Wcur   += w[city]
                if Wcur > Wmax:
                    return -1e18  # infeasible = super bad

            v = v_max - (v_max - v_min) * (Wcur / float(Wmax))
            v = max(v_min, min(v, v_max))

            time += d[city, nxt] / v

        return profit - alpha * time

    # -----------------------------------------------------------
    #   MAIN VND CALL (GA-compatible)
    # -----------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **call_configs):
        """
        offspring = {"tsp": route, "kp": kp_bits}
        Returns NEW improved offspring dict.
        """
        cfg = dict(self.__configs)
        cfg.update(call_configs)
        max_rounds = cfg.get("max_rounds", self.max_rounds)

        # Normalize "offspring" into dict form (tsp, kp)
        if isinstance(offspring, dict):
            tsp = np.array(offspring["tsp"], copy=True)
            kp  = np.array(offspring["kp"],  copy=True)
        else:
            # offspring is a structured genome row (Genoms object)
            tsp = np.array(offspring["tsp"], copy=True)
            kp  = np.array(offspring["kp"],  copy=True)

        # Work with (tsp, kp) from here


        # optional KP greedy improve at start
        if self.use_kp_ls:
            tmp = {"tsp": tsp, "kp": kp}
            tmp = self.kp_ls(None, None, tmp)
            tsp = np.array(tmp["tsp"], copy=True)
            kp  = np.array(tmp["kp"],  copy=True)

        cur_route = tsp.copy()
        cur_kp    = kp.copy()
        cur_score = self._compute_ttp_score(cur_route, cur_kp)

        # ---------------- VND LOOP ------------------
        for _ in range(max_rounds):
            improved = False

            for op in self.ops:
                # 1) route LS
                cand_route = op(parent1, parent2, cur_route)
                cand_kp    = cur_kp.copy()

                # 2) KP LS after route change
                if self.use_kp_ls:
                    tmp = {"tsp": cand_route, "kp": cand_kp}
                    tmp = self.kp_ls(None, None, tmp)
                    cand_route = np.array(tmp["tsp"], copy=True)
                    cand_kp    = np.array(tmp["kp"],  copy=True)

                # 3) TTP score
                cand_score = self._compute_ttp_score(cand_route, cand_kp)

                if cand_score > cur_score + 1e-9:
                    cur_route = cand_route
                    cur_kp    = cand_kp
                    cur_score = cand_score
                    improved  = True

            if not improved:
                break

        # Return same type as input
        if isinstance(offspring, dict):
            return {"tsp": cur_route, "kp": cur_kp}
        else:
            out = offspring.copy()
            out["tsp"] = cur_route
            out["kp"]  = cur_kp
            return out
