#!/usr/bin/env python3
"""Plot the tight-bound H4 failure-recognition outcomes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "results" / "cross_setting"


def main() -> None:
    data = pd.read_csv(ROOT / "h4_failure_recognition.csv")
    data = data[data["ratio"] == 1.0].copy()
    labels = {
        ("bloodmnist", "adaptive_patch", "spectral_signatures"):
            "BloodMNIST\nAdaptive-Patch / Spectral",
        ("bloodmnist", "blended", "cf_cmd"):
            "BloodMNIST\nBlended / CF-CMD input",
        ("cifar10", "badnets", "target_centroid_distance"):
            "CIFAR-10\nBadNets / Target centroid",
    }
    data["label"] = [labels[(r.dataset, r.attack, r.method)] for r in data.itertuples()]
    order = [
        "BloodMNIST\nAdaptive-Patch / Spectral",
        "BloodMNIST\nBlended / CF-CMD input",
        "CIFAR-10\nBadNets / Target centroid",
    ]
    data["label"] = pd.Categorical(data["label"], categories=order, ordered=True)
    data = data.sort_values("label")

    go = 100 * data["go_rate"].to_numpy()
    nogo = 100 * data["nogo_rate"].to_numpy()
    abstain = 100 * data["abstain_rate"].to_numpy()
    false_go = 100 * data["false_go_rate"].to_numpy()
    y = np.arange(len(data))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.18, top=0.72)
    colors = {"go": "#0072B2", "nogo": "#D55E00", "abstain": "#B8B8B8"}
    ax.barh(y, go, color=colors["go"], label="GO")
    ax.barh(y, nogo, left=go, color=colors["nogo"], label="NO-GO")
    ax.barh(y, abstain, left=go + nogo, color=colors["abstain"], label="Abstain")
    ax.scatter(false_go, y, marker="D", s=42, facecolor="white", edgecolor="black",
               linewidth=1.2, zorder=4, label="False GO")
    for yi, fg in zip(y, false_go):
        ax.annotate(f"{fg:.2f}%", (fg, yi), xytext=(5, -11), textcoords="offset points",
                    ha="left", va="center", fontsize=9)
    ax.axvline(5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(5.7, 1.02, r"$\delta=5\%$", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=9)
    ax.set_yticks(y, data["label"].astype(str))
    ax.set_xlim(-1.5, 100)
    ax.set_xlabel("Decision frequency across trusted-set redraws (%)")
    ax.set_title("Failure recognition with the amended tight clean-rate bound", pad=14)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    handles, legend_labels = ax.get_legend_handles_labels()
    legend_order = [legend_labels.index(name) for name in ("GO", "NO-GO", "Abstain", "False GO")]
    ax.legend([handles[index] for index in legend_order],
              [legend_labels[index] for index in legend_order],
              loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.savefig(ROOT / "failure_recognition.png", dpi=240)
    fig.savefig(ROOT / "failure_recognition.pdf")
    print(ROOT / "failure_recognition.png")


if __name__ == "__main__":
    main()
