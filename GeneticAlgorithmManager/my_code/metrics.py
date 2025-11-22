#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules

sys_remove_modules("extern_fn")
from extern_fn import *

class Metrics(ExtenFn):
    """
    Wrapper for external metric modules.
    Automatically detects MetricsTTP.getScoreTTP.
    """

    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Metrics")
        self._extern_fn = self.__unpack(extern_fn)

    def __call__(self, genomics):
        return self._extern_fn(genomics)

    def __unpack(self, extern_fn):
        fn = self.metricsAbstract
        self.getScore = self.getScoreAbstract

        if extern_fn is not None:
            fn = extern_fn

            # Prefer special TTP scoring
            if hasattr(extern_fn, "getScoreTTP"):
                self.getScore = extern_fn.getScoreTTP

            # Fallback to normal getScore if exists
            elif hasattr(extern_fn, "getScore"):
                self.getScore = extern_fn.getScore

        return fn

    def __getattr__(self, name):
        """
        Forward any unknown attribute to wrapped external metrics class:
           - metrics_cache
           - getArgBest
           - computeIndividDistance
           - computeNbrObjKP
           - etc.
        """
        return getattr(self._extern_fn, name)

    def metricsAbstract(self, *args):
        raise NameError(
            f"Functia 'Metrics', lipseste functia externa '{self._extern_fn}'"
        )

    def getScoreAbstract(self, *args):
        raise NameError(
            f"Functia 'Metrics', lipseste functia 'getScore' din extern '{self._extern_fn}'"
        )
