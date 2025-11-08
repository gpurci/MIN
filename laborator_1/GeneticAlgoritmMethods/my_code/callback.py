#!/usr/bin/python

import pandas as pd
from pathlib import Path

class Callback(object):
    """
    Salveaza rezultatele obtinute pentru fiecare generatie in 'csv' file.
    """

    def __init__(self, filename):
        self.filename   = filename
        self.pd_history = None
        self.epoch = 0
        if (Path(self.filename).is_file()):
            self.pd_history = pd.read_csv(self.filename)
            is_epoch = self.pd_history.get("epoch", None)
            if (is_epoch is not None):
                self.epoch = self.pd_history.at[len(self.pd_history)-1, "epoch"]


    def __call__(self, epoch, logs):
        # valorile de pe 'key' trebuie sa fie liste sau vector
        tmp_logs = logs.copy()
        tmp_logs["epoch"] = epoch+self.epoch
        for key in tmp_logs.keys():
            val = [tmp_logs[key]]
            tmp_logs[key] = val
        # salveaza logurile in data frame
        pd_df = pd.DataFrame(data=tmp_logs)
        # adauga logurile in lista de loguri
        if (self.pd_history is None):
            self.pd_history = pd_df
        else:
            self.pd_history = pd.concat([self.pd_history, pd_df], ignore_index=True)
        # salveaza in 'csv' file
        self.pd_history.to_csv(self.filename, index=False) 

    def help(self):
        info = """Callback: 
        metode de config: 'filename'\n"""
        return info
