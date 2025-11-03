#!/usr/bin/python

from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd


class KPGenerator(object):
    def __init__(self):
        pass

    def read_csv(self, filename):
        pd_data = pd.read_csv(filename, sep="\t", header=0)
        return pd_data

