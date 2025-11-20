#!/usr/bin/python

import pandas as pd
from pathlib import Path
import warnings

class Callback(object):
    """
    Salveaza rezultatele obtinute pentru fiecare generatie in 'csv' file.
    """
    def __init__(self, filename="", freq=1):
        self.filename   = filename
        self.pd_history = None
        self.epoch = 0
        self.freq  = freq
        # daca este fisierul se actualizeaza valoarea epocii
        if (isinstance(self.filename, str)):
            if (Path(self.filename).is_file()):
                self.pd_history = pd.read_csv(self.filename)
                is_epoch = self.pd_history.get("epoch", None)
                if (is_epoch is not None):
                    self.epoch = self.pd_history.at[len(self.pd_history)-1, "epoch"]
            else:
                path = Path(self.filename).parent
                Path(path).mkdir(mode=0o777, parents=True, exist_ok=True)
                Path(self.filename).touch(mode=0o666, exist_ok=True)
        else:
            warnings.warn("\nCallback: Numele fisierului '{}' este type '{}'\n".format(self.filename, type(self.filename)))


    def __str__(self):
        info = "Callback: filename {}".format(self.filename)
        return info

    def __call__(self, epoch, logs):
        # salvare cu o frecventa
        if ((epoch % self.freq) == 0):
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
        info = """Callback: "filename":filename, "freq":1\n"""
        return info
