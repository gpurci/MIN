#!/usr/bin/python

import numpy as np
import sys

def sys_remove_modules(module_name, *args):
    if (module_name in sys.modules):
        del sys.modules[module_name]
        for key in args:
            tmp_modules = "{}.{}".format(module_name, key)
            del sys.modules[tmp_modules]
