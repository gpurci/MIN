#!/usr/bin/python

import numpy as np
#import traceback
from root_GA import *

class ExtenFn():
    """
    Clasa 'ExtenFn', 
    """
    def __init__(self, extern_fn=None, name=""):
        super().__init__()
        self._extern_fn = self.__unpack(extern_fn)
        self.__name      = name

    def __call__(self, *args):
        raise NameError("Functia '{}', lipseste implementarea: '__call__'".format(self.__name))

    def __str__(self):
        info = "{}: ".format(self.__name)
        if (self._extern_fn is not None):
            info += "{}".format(str(self._extern_fn))
        else:
            info += "{}".format(None)
        return info

    def __unpack(self, extern_fn):
        fn = self.abstract
        if (extern_fn is not None):
            fn = extern_fn
        return fn

    def help(self):
        info = """{}: 'extern'; config: 'extern_kw';\n""".format(self.__name)
        return info

    def setParameters(self, **kw):
        if (self._extern_fn is not None):
            if (issubclass(self._extern_fn, RootGA)):
                self._extern_fn.setParameters(**kw)
            else:
                raise NameError("Functia '{}', functia externa '{}', nu mosteneste 'RootGA'".format(self.__name, self._extern_fn))
        else:
            raise NameError("Functia '{}', lipseste functia externa '{}'".format(self.__name, self._extern_fn))

    def abstract(self, *args):
        raise NameError("Functia '{}', lipseste functia externa '{}'".format(self.__name, self._extern_fn))
