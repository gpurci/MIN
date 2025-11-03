#!/usr/bin/python

from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd


class TTPGenerator(object):
    def __init__(self, nbr_city, max_distance, coord=None):
        self.nbr_city     = nbr_city
        self.max_distance = max_distance
        if (coord is not None):
            self.coord = coord
        else:
            self.coord = np.random.randint(low=0, high=self.max_distance, size=(self.nbr_city, 2))

    def __call__(self):
        map_of_distance = []
        for point in self.coord:
            tmp_distance = np.linalg.norm(self.coord - point, axis=1)
            #print(tmp_distance)
            map_of_distance.append(tmp_distance)
        return np.array(map_of_distance)

    def __put_points(self, map):
        for x, y in self.coord:
            cv.circle(map, (x, y), 4, (0, 255, 0), -1)

    def putRoutesOnMap(self, routes):
        map = np.zeros((self.max_distance, self.max_distance, 3), dtype=np.uint8)
        self.__put_points(map)

        st_x, st_y = self.coord[routes[0]]
        cv.circle(map, (st_x, st_y), 5, (255, 255, 255), 1)
        for route in routes[1:]:
            en_x, en_y = self.coord[route]
            cv.circle(map, (st_x, st_y), 4, (0, 0, 255), -1)
            cv.line(map,   (st_x, st_y), (en_x, en_y), (255, 0, 0), 4)
            st_x, st_y = en_x, en_y
        cv.circle(map, (st_x, st_y), 8, (0, 255, 255), 1)
        return map

    def read_csv(self, filename):
        return pd.read_csv(filename, sep="\t", header=0)

