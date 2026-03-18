import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_survival(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    max_steps = int(df["max_steps"].max())
    plt.figure(figsize=(8, 5))
    for group, g in df.groupby("group"):
        times = []
        survival = []
        for t in range(max_steps + 1):
            survived = ((g["t_collapse"] == -1) | (g["t_collapse"] > t)).mean()
            times.append(t)
            survival.append(survived)
        plt.step(times, survival, where="post", label=group)
    plt.xlabel("Generation step")
    plt.ylabel("Survival probability")
    plt.title("Collapse Survival Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
