#!/usr/bin/python

from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd


class TSPGenerator(object):
    def __init__(self):
        self.coords = None

    def __call__(self):
        map_of_distance = []
        for point in self.coords:
            tmp_distance = np.linalg.norm(self.coords - point, axis=1)
            #print(tmp_distance)
            map_of_distance.append(tmp_distance)
        # update dataset
        self.dataset = {
        "GENOME_LENGTH":map_of_distance.shape[0],
        "distance":     map_of_distance}
        return self.dataset

    def __put_points(self, map):
        for x, y in self.coords:
            cv.circle(map, (x, y), 4, (0, 255, 0), -1)

    def putRoutesOnMap(self, routes):
        map = np.zeros((self.max_distance, self.max_distance, 3), dtype=np.uint8)
        self.__put_points(map)

        st_x, st_y = self.coords[routes[0]]
        cv.circle(map, (st_x, st_y), 5, (255, 255, 255), 1)
        for route in routes[1:]:
            en_x, en_y = self.coords[route]
            cv.circle(map, (st_x, st_y), 4, (0, 0, 255), -1)
            cv.line(map,   (st_x, st_y), (en_x, en_y), (255, 0, 0), 4)
            st_x, st_y = en_x, en_y
        cv.circle(map, (st_x, st_y), 8, (0, 255, 255), 1)
        return map

    def read_csv(self, filename):
        pd_data = pd.read_csv(filename, sep="\t", header=0)
        x = np.array(pd_data["X"], dtype=np.int32)
        y = np.array(pd_data["Y"], dtype=np.int32)
        self.coords = np.array(list(zip(x, y)), dtype=np.int32)
        self.max_distance = self.coords.max() + 10
        return pd_data

