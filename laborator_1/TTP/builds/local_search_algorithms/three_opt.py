#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class ThreeOpt(RootGA):
    """
    Restricted 3-opt for TSP routes.
    Samples (i,j,k) triplets and tries common 3-opt reconnections.

    Method:
        "three_opt_restrict"

    Configs:
        iters=60
        k_candidates=15
        sample_i=30
        seed=None
    """

    def __init__(self, method="three_opt_restrict", dataset=None, **configs):
        super().__init__()
        if dataset is None:
            raise ValueError("ThreeOpt requires dataset with 'distance'.")
        self.dataset = dataset
        self.distance = dataset["distance"]
        self.__method = method
        self.__configs = configs
        self.__fn = self.__unpack(method)
        self._candidates = None

    def __str__(self):
        return f"ThreeOpt(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""ThreeOpt:
    method: 'three_opt_restrict'
    configs:
        iters=60, k_candidates=15, sample_i=30, seed=None
""")

    def __unpack(self, method):
        table = {
            "three_opt_restrict": self.threeOptRestrict
        }
        return table.get(method, self._abstract)

    def _abstract(self, *a, **kw):
        raise NameError(f"Missing method '{self.__method}' in ThreeOpt")

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if self._candidates is None:
            k = self.__configs.get("k_candidates", 15)
            self._build_candidate_lists(k)

    def __call__(self, p1, p2, route, **call_configs):
        cfg = dict(self.__configs)
        cfg.update(call_configs)
        return self.__fn(p1, p2, route, **cfg)

    # ---------- helpers ----------
    def _build_candidate_lists(self, k=15):
        n = self.GENOME_LENGTH
        D = self.distance
        cand = []
        for c in range(n):
            nn = np.argsort(D[c])[1:k+1]
            cand.append(nn.astype(np.int32))
        self._candidates = cand

    def _route_length(self, r):
        return self.distance[r[-1], r[0]] + np.sum(self.distance[r[:-1], r[1:]])

    def _best_3opt_reconnect(self, r, i, j, k):
        """
        Try standard 3-opt reconnections for i<j<k on linear indices.
        Returns best new route and its length.
        """
        A = r[:i+1]
        B = r[i+1:j+1]
        C = r[j+1:k+1]
        D = r[k+1:]

        cands = [
            np.concatenate([A, B[::-1], C, D]),            # case1
            np.concatenate([A, B, C[::-1], D]),            # case2
            np.concatenate([A, B[::-1], C[::-1], D]),      # case3
            np.concatenate([A, C, B, D]),                  # case4
            np.concatenate([A, C[::-1], B, D]),            # case5
            np.concatenate([A, C, B[::-1], D]),            # case6
            np.concatenate([A, C[::-1], B[::-1], D]),      # case7
        ]

        best_r = r
        best_len = self._route_length(r)
        for rr in cands:
            ll = self._route_length(rr)
            if ll < best_len:
                best_len = ll
                best_r = rr
        return best_r, best_len

    # ---------- method ----------
    def threeOptRestrict(self, p1, p2, route,
                        iters=60, k_candidates=15, sample_i=30, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self._candidates is None:
            self._build_candidate_lists(k_candidates)

        cur = route.copy()
        cur_len = self._route_length(cur)
        n = len(cur)

        pos = np.empty(n, dtype=np.int32)
        pos[cur] = np.arange(n)

        for _ in range(iters):
            improved = False

            i_list = np.random.choice(np.arange(1, n-4), size=min(sample_i, n-5), replace=False)
            for i in i_list:
                a = cur[i]
                for c in self._candidates[a]:
                    j = pos[c]
                    if not (i+1 < j < n-2):
                        continue
                    # pick k from neighbors of city at j
                    b = cur[j]
                    for e in self._candidates[b]:
                        k = pos[e]
                        if not (j+1 < k < n-1):
                            continue

                        new_r, new_len = self._best_3opt_reconnect(cur, i, j, k)
                        if new_len + 1e-9 < cur_len:
                            cur = new_r
                            cur_len = new_len
                            pos[cur] = np.arange(n)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return cur
