#!/usr/bin/python

import numpy as np
import cv2 as cv
import pandas as pd

class TTPGenerator(object):
    def __init__(self, path):
        self.path         = path
        self.max_distance = 0
        self.dataset      = None
        self.coords       = None

    def __call__(self, nodes_file: str, items_file: str):
        # citeste coordonatele oraselor
        df_nodes = self.read_csv(nodes_file)
        x = np.array(df_nodes["X"], dtype=np.int32)
        y = np.array(df_nodes["Y"], dtype=np.int32)
        self.coords = np.array(list(zip(x, y)), dtype=np.int32)
        self.max_distance = int(self.coords.max()) + 10
        # calculeaza distatele
        distance = self._pairwise_distance(is_ceil2d=True)
        # citeste greutatile si profitul
        df_items = self.read_csv(items_file)
        # initializare
        prof = np.zeros(self.coords.shape[0], dtype=float)
        wgt  = np.zeros(self.coords.shape[0], dtype=float)
        # atribuire profit si greutate, pozitiilor din care fac parte
        for _, row in df_items.iterrows():
            city = int(row["ASSIGNED_NODE_NUMBER"]) - 1
            prof[city] = float(row["PROFIT"])
            wgt [city] = float(row["WEIGHT"])
        # update dataset
        self.dataset = {
        "distance":    distance,
        "coords":      self.coords,
        "item_profit": prof,
        "item_weight": wgt}
        return self.dataset

    def __put_points(self, image):
        for x, y in self.coords:
            cv.circle(image, (x, y), 4, (0, 255, 0), -1)

    def putRoutesOnMap(self, routes):
        image = np.zeros((self.max_distance, self.max_distance, 3), dtype=np.uint8)
        self.__put_points(image)

        st_x, st_y = self.coords[routes[0]]
        cv.circle(image, (st_x, st_y), 5, (255, 255, 255), 1)
        for route in routes[1:]:
            en_x, en_y = self.coords[route]
            cv.circle(image, (st_x, st_y), 4, (0, 0, 255), -1)
            cv.line(image,   (st_x, st_y), (en_x, en_y), (255, 0, 0), 4)
            st_x, st_y = en_x, en_y
        cv.circle(image, (st_x, st_y), 8, (0, 255, 255), 1)
        return image

    def read_csv(self, filename):
        filename = "{}/{}".format(self.path, filename)
        return pd.read_csv(filename, sep="\t", header=0)
    
    # calculeaza distante perechi, aplicand CEIL_2D (rotunjire în sus, nu distanta euclidiana reala)
    def _pairwise_distance(self, is_ceil2d: bool = True) -> np.ndarray:
        map_of_distance = []
        for point in self.coords:
            tmp_distance = np.linalg.norm(self.coords - point, axis=1)
            #print(tmp_distance)
            map_of_distance.append(tmp_distance)
        map_of_distance = np.array(map_of_distance)
        if (is_ceil2d):
            map_of_distance = np.round(map_of_distance, 0)
        return map_of_distance
