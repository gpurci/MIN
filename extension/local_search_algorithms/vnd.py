#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA

# import your local_search operators
from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.three_opt import ThreeOpt


class VND(RootGA):
    """
    Variable Neighborhood Descent for TSP routes.
    Applies a sequence of local searches repeatedly until no improvement.

    Default neighborhoods:
        1) TwoOpt("two_opt_distance")
        2) OrOpt("or_opt_restrict")
        3) ThreeOpt("three_opt_restrict")

    Configs:
        max_rounds=5
        seed=None
    """

    def __init__(self, dataset=None, ops=None, **configs):
        super().__init__()
        if dataset is None:
            raise ValueError("VND requires dataset.")
        self.dataset = dataset
        self.distance = dataset["distance"]
        self.__configs = configs

        if ops is None:
            ops = [
                TwoOpt("two_opt_distance", dataset),
                OrOpt("or_opt_restrict", dataset),
                ThreeOpt("three_opt_restrict", dataset),
            ]
        self.ops = ops

    def __str__(self):
        return f"VND(ops={[op.__class__.__name__ for op in self.ops]}, configs={self.__configs})"

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for op in self.ops:
            if hasattr(op, "setParameters"):
                op.setParameters(**kw)

    def __call__(self, p1, p2, route, **call_configs):
        cfg = dict(self.__configs)
        cfg.update(call_configs)

        max_rounds = cfg.get("max_rounds", 5)
        seed = cfg.get("seed", None)
        if seed is not None:
            np.random.seed(seed)

        def route_len(r):
            return self.distance[r[-1], r[0]] + np.sum(self.distance[r[:-1], r[1:]])

        cur = route.copy()
        cur_len = route_len(cur)

        for _ in range(max_rounds):
            improved = False
            for op in self.ops:
                new_r = op(None, None, cur)
                new_len = route_len(new_r)
                if new_len + 1e-9 < cur_len:
                    cur = new_r
                    cur_len = new_len
                    improved = True
            if not improved:
                break

        return cur
