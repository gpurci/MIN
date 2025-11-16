#!/usr/bin/python
import pandas as pd
from pathlib import Path
import warnings

class Callback(object):
    """
    Salveaza rezultatele pentru fiecare generatie in fisierul CSV.
    """

    def __init__(self, filename="", freq=1):

        self.filename = filename
        self.freq = freq

        self.pd_history = None
        self.base_epoch = 0  # offset pentru continuare

        if isinstance(filename, str):
            path = Path(filename)

            if path.is_file():
                # Încarcă istoricul
                self.pd_history = pd.read_csv(path)

                if "Generatia" in self.pd_history.columns:
                    # ultima generație salvată
                    self.base_epoch = int(self.pd_history["Generatia"].iloc[-1]) + 1

            else:
                # creează folderul
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(mode=0o666, exist_ok=True)

        else:
            warnings.warn(f"Callback filename must be str, got {type(filename)}")

    # ----------------------------------------------------------------------
    def __call__(self, epoch, logs):

        # salvează doar la fiecare freq generații
        if (epoch % self.freq) != 0:
            return

        # adaugă generația reală
        row = {"Epoch": self.base_epoch + epoch}

        # copiază toate metricele (score, profit, distance, time, weight…)
        row.update(logs)

        # Adaugă în DataFrame
        df_row = pd.DataFrame([row])   # o singură linie

        if self.pd_history is None:
            self.pd_history = df_row
        else:
            self.pd_history = pd.concat([self.pd_history, df_row], ignore_index=True)

        # Scrie în CSV
        self.pd_history.to_csv(self.filename, index=False)

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"Callback: filename={self.filename}"
