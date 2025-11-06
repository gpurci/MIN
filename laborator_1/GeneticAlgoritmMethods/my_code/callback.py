#!/usr/bin/python

import pandas as pd

class Callback(object):
    """
    Salveaza rezultatele obtinute pentru fiecare generatie in 'csv' file.
    """

    def __init__(self, filename):
        self.filename   = filename
        self.pd_history = None

    def __call__(self, epoch, logs):
        # valorile de pe 'key' trebuie sa fie liste sau vector
        for key in logs.keys():
            val = [logs[key]]
            logs[key] = val
        # salveaza logurile in data frame
        pd_df = pd.DataFrame(data=logs)
        # adauga logurile in lista de loguri
        if (self.pd_history is None):
            self.pd_history = pd_df
        else:
            self.pd_history.append(pd_df, ignore_index=True)
        # salveaza in 'csv' file
        self.pd_history.to_csv(self.filename, index=False) 


