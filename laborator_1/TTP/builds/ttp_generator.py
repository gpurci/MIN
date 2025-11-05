#!/usr/bin/python

from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
import csv
from math import hypot, ceil

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
    
    def load_ttp_csv(self, nodes_csv_path: str, items_csv_path: str):
        df_nodes = pd.read_csv(nodes_csv_path, sep="\t")
        coords = df_nodes[["X","Y"]].to_numpy(dtype=float)

        n = coords.shape[0]
        distance = self._pairwise_distance(coords, ceil2d=True)

        df_items = pd.read_csv(items_csv_path, sep="\t")
        prof = np.zeros(n, dtype=float)
        wgt  = np.zeros(n, dtype=float)

        for _, row in df_items.iterrows():
            city = int(row["ASSIGNED_NODE_NUMBER"]) - 1
            prof[city] = float(row["PROFIT"])
            wgt [city] = float(row["WEIGHT"])

        self.coords      = coords
        self.distance    = distance
        self.item_profit = prof
        self.item_weight = wgt


    # utilitar numeric
    def _is_number(self, x) -> bool:
        try:
            float(x); return True
        except:
            return False

    # calculeaza distante perechi, aplicand CEIL_2D (rotunjire în sus, nu distanta euclidiana reala)
    def _pairwise_distance(self, coords: np.ndarray, ceil2d: bool = True) -> np.ndarray:
        n = coords.shape[0]
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i+1, n):
                xj, yj = coords[j]
                d = hypot(xi - xj, yi - yj)
                D[i, j] = D[j, i] = float(ceil(d)) if ceil2d else d
        return D