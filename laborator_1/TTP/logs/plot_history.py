#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


def plot_history(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded:", df.shape)

    # Determine which column to use as epoch
    possible_epoch_cols = ["epoch", "Epoch", "generatia", "Generatia"]
    xcol = None

    for col in possible_epoch_cols:
        if col in df.columns:
            xcol = col
            break

    if xcol is None:
        raise KeyError(
            "CSV must contain one of these columns: epoch, Epoch, Generatia"
        )

    print("Using X axis:", xcol)

    folder = Path("plots")
    folder.mkdir(exist_ok=True)

    pdf = PdfPages(folder / "report.pdf")

    # Which metrics to plot
    metrics = ["score", "profit", "distance", "time", "weight", "best_fitness"]

    for y in metrics:
        if y not in df.columns:
            print(f"Skipping (missing): {y}")
            continue

        plt.figure()
        plt.plot(df[xcol], df[y])
        plt.xlabel(xcol)
        plt.ylabel(y)
        plt.title(f"{y} evolution")
        plt.grid()

        out_path = folder / f"{y}.png"
        plt.savefig(out_path, dpi=150)
        pdf.savefig()
        plt.close()

    pdf.close()
    print("All plots saved in:", folder)


if __name__ == "__main__":
    plot_history("history.csv")
