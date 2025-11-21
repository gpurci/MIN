class TabuRoute:
    """Wrapper that makes a full TabuSearch operator callable on a single TSP route."""

    def __init__(self, tabu_obj, method="tabu_search_distance"):
        self.tabu = tabu_obj
        self.method = method

    def setParameters(self, **kw):
        self.tabu.setParameters(**kw)

    def __call__(self, route):
        # Apply tabu search to a single TSP route.
        if self.method == "tabu_search":
            return self.tabu.tabuSearch(None, None, route)
        if self.method == "tabu_search_rand":
            return self.tabu.tabuSearchRand(None, None, route)
        return self.tabu.tabuSearchDistance(None, None, route)
