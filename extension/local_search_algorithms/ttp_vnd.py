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
        **configs,
    ):
        super().__init__()

        if dataset is None:
            raise ValueError("TTPVNDLocalSearch requires dataset")

        self.dataset = dataset
        self.distance = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        self.v_max = v_max
        self.v_min = v_min
        self.Wmax = W
        self.alpha = alpha
        self.max_rounds = max_rounds
        self.use_kp_ls = use_kp_ls

        # keep any extra configs (e.g. "repair") for later use
        self.__configs = dict(configs)

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

    def help(self):
        info = "TTPVNDLocalSearch: full TTP-aware VND (2-opt, or-opt, 3-opt + KP LS)\n"
        info += "  use_kp_ls: use greedy KP LS inside VND (bool)\n"
        info += "  max_rounds: max VND passes over neighborhood list (int)\n"
        print(info)

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
        Wmax = self.Wmax
        alpha = self.alpha

        Wcur = 0.0
        time = 0.0
        profit = 0.0
        n = len(route)

        for idx in range(n):
            city = route[idx]
            nxt = route[(idx + 1) % n]

            # pick item
            if kp[city] == 1:
                profit += p[city]
                Wcur += w[city]
                if Wcur > Wmax:
                    # infeasible = very bad
                    return -1e18

            # clamp speed
            v = v_max - (v_max - v_min) * (Wcur / float(Wmax))
            if v < v_min:
                v = v_min
            elif v > v_max:
                v = v_max

            time += d[city, nxt] / v

        return profit - alpha * time

    # -----------------------------------------------------------
    #   Diversification "shake" when VND is stuck
    # -----------------------------------------------------------
    def _shake(self, route, kp, max_k=6):
        """
        Small random perturbation of the route (and optional KP re-LS).
        Used only when no improving neighbor is found.
        """
        n = len(route)
        if n < 4:
            return route, kp

        # choose a random subset of positions and permute them
        k = np.random.randint(2, min(max_k, n))
        idx = np.random.choice(n, size=k, replace=False)
        new_route = route.copy()
        new_route[idx] = new_route[idx[np.random.permutation(k)]]

        new_kp = kp.copy()

        # optionally re-run KP LS after shake to keep feasibility/quality
        if self.use_kp_ls:
            tmp = {"tsp": new_route, "kp": new_kp}
            tmp = self.kp_ls(None, None, tmp)
            new_route = np.array(tmp["tsp"], copy=True)
            new_kp = np.array(tmp["kp"], copy=True)

        return new_route, new_kp

    # -----------------------------------------------------------
    #   MAIN VND CALL (GA-compatible)
    # -----------------------------------------------------------
    def __call__(self, parent1, parent2, offspring, **call_configs):
        """
        offspring = {"tsp": route, "kp": kp_bits}
        Returns NEW improved offspring dict or structured genome.
        """
        cfg = dict(self.__configs)
        cfg.update(call_configs)
        max_rounds = cfg.get("max_rounds", self.max_rounds)

        # Normalize "offspring" into dict form (tsp, kp)
        if isinstance(offspring, dict):
            tsp = np.array(offspring["tsp"], copy=True)
            kp = np.array(offspring["kp"], copy=True)
            out_type_dict = True
        else:
            # offspring is a structured genome row (Genoms object)
            tsp = np.array(offspring["tsp"], copy=True)
            kp = np.array(offspring["kp"], copy=True)
            out_type_dict = False

        # optional KP greedy improve at start
        if self.use_kp_ls:
            tmp = {"tsp": tsp, "kp": kp}
            tmp = self.kp_ls(None, None, tmp)
            tsp = np.array(tmp["tsp"], copy=True)
            kp = np.array(tmp["kp"], copy=True)

        cur_route = tsp.copy()
        cur_kp = kp.copy()
        cur_score = self._compute_ttp_score(cur_route, cur_kp)

        # ---------------- VND LOOP ------------------
        for _ in range(max_rounds):
            improved = False

            for op in self.ops:
                # 1) route LS
                cand_route = op(parent1, parent2, cur_route)
                cand_kp = cur_kp.copy()

                # 2) KP LS after route change
                if self.use_kp_ls:
                    tmp = {"tsp": cand_route, "kp": cand_kp}
                    tmp = self.kp_ls(None, None, tmp)
                    cand_route = np.array(tmp["tsp"], copy=True)
                    cand_kp = np.array(tmp["kp"], copy=True)

                # 3) TTP score
                cand_score = self._compute_ttp_score(cand_route, cand_kp)

                if cand_score > cur_score + 1e-9:
                    cur_route = cand_route
                    cur_kp = cand_kp
                    cur_score = cand_score
                    improved = True

            # If nothing improved in this whole pass, try a diversification shake
            if not improved:
                shake_route, shake_kp = self._shake(cur_route, cur_kp)
                shake_score = self._compute_ttp_score(shake_route, shake_kp)

                if shake_score > cur_score + 1e-9:
                    # accept shaken solution and continue VND from there
                    cur_route = shake_route
                    cur_kp = shake_kp
                    cur_score = shake_score
                else:
                    # still stuck: stop VND
                    break

        # Return same type as input
        if out_type_dict:
            return {"tsp": cur_route, "kp": cur_kp}
        else:
            out = offspring.copy()
            out["tsp"] = cur_route
            out["kp"] = cur_kp
            return out
