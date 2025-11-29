#!/usr/bin/python

import numpy as np
import sys
import importlib

def sys_remove_modules(module_name, *args):
    if (module_name in sys.modules):
        if (module_name in sys.modules):
            del sys.modules[module_name]
        for key in args:
            tmp_modules = "{}.{}".format(module_name, key)
            if (tmp_modules in sys.modules):
                del sys.modules[tmp_modules]

def sys_reaload_module(str_module):
    if (str_module in sys.modules):
        importlib.reload(sys.modules[str_module])
    else:
        importlib.import_module(str_module)
