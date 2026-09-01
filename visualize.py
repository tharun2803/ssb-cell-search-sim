"""
Plotting helpers. All functions save a PNG to `out_path` and return
the matplotlib Figure (in case the caller wants to display it too,
e.g. in a notebook).
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script/CI use
import matplotlib.pyplot as plt
import numpy as np


def plot_pss_correlation_peaks(corr_curves: dict, true_offset: int,
                                true_n_id_2: int, out_path: str):
    """
    Plot normalised PSS correlation magnitude vs. lag, for all 3
    candidate N_ID_2 sequences, with the true symbol start marked.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for n_id_2, curve in corr_curves.items():
        style = "-" if n_id_2 == true_n_id_2 else "--"
        ax.plot(curve, style, color=colors.get(n_id_2, None),
                 label=f"N_ID_2 = {n_id_2}" + (" (true)" if n_id_2 == true_n_id_2 else ""))

    ax.axvline(true_offset, color="red", linestyle=":", linewidth=1.5,
               label="True PSS symbol start")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Normalised correlation magnitude")
    ax.set_title("PSS sliding correlation vs. lag (cell-search timing acquisition)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_detection_vs_snr(df, out_path: str):
    """Plot P(correct full cell-ID) and P(correct N_ID_2) vs SNR."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["snr_db"], df["detection_prob"], "o-", label="Full N_ID_cell correct")
    ax.plot(df["snr_db"], df["pss_only_detection_prob"], "s--",
            label="N_ID_2 correct (PSS stage only)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection probability")
    ax.set_title("Cell-search detection reliability vs. SNR")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_search_time_vs_snr(df, out_path: str):
    """Plot the estimated expected cell-search time vs SNR."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["snr_db"], df["expected_search_time_ms"], "o-", color="tab:red")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Expected cell-search time (ms)")
    ax.set_title("Estimated cell-search time vs. SNR")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig
