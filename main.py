"""
Main entry point for the SS/PBCH Block (SSB) cell-search simulator.

Running this script produces all the project deliverables:
    1. results/plots/pss_correlation_peaks.png   - correlation-peak visualisation
       for one example (single-trial) cell search
    2. results/detection_vs_snr.csv               - Monte Carlo results table
    3. results/plots/detection_vs_snr.png         - detection reliability vs SNR
    4. results/plots/search_time_vs_snr.png       - estimated cell-search time vs SNR
    5. Console summary of a single example cell search

Usage:
    python main.py
    python main.py --trials 500 --snr -10 -8 -6 -4 -2 0 2 4 6 8 10
"""

import argparse
import os

import numpy as np

from ssb_sim import channel, simulate, visualize
from ssb_sim.ssb import build_ssb_waveform, SYMBOL_LEN
from ssb_sim.detector import full_cell_search
from ssb_sim.sss import cell_id

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def run_example_single_search(snr_db: float, seed: int = 7):
    """Run one illustrative cell search and plot its PSS correlation curves."""
    rng = np.random.default_rng(seed)
    true_n_id_1, true_n_id_2 = 200, 1  # arbitrary example cell identity
    true_cell = cell_id(true_n_id_1, true_n_id_2)

    ssb_wave = build_ssb_waveform(true_n_id_1, true_n_id_2)
    window_len = 4 * SYMBOL_LEN + simulate.SEARCH_WINDOW_MARGIN
    window, true_offset = channel.embed_in_search_window(ssb_wave, window_len, rng)
    rx = channel.awgn(window, snr_db, rng)

    result = full_cell_search(rx)

    print("=" * 60)
    print(f"EXAMPLE SINGLE CELL SEARCH  (SNR = {snr_db} dB)")
    print("=" * 60)
    print(f"  True cell:       N_ID_1={true_n_id_1}, N_ID_2={true_n_id_2}, "
          f"N_ID_cell={true_cell}")
    print(f"  True PSS offset: {true_offset} (post-CP sample index)")
    print("-" * 60)
    print(f"  Detected N_ID_2:   {result['n_id_2']}  "
          f"(PSS peak={result['pss_peak']:.3f}, offset={result['pss_offset']})")
    print(f"  Detected N_ID_1:   {result['n_id_1']}  (SSS score={result['sss_score']:.3f})")
    print(f"  Detected cell ID:  {result['n_id_cell']}")
    print(f"  Correct?           {result['n_id_cell'] == true_cell}")
    print("=" * 60)

    # The detector's "offset" is the start of the *useful* (post-CP) PSS
    # symbol, so mark true_offset + CP_LEN on the plot to match it exactly.
    from ssb_sim import ofdm
    true_useful_offset = true_offset + ofdm.CP_LEN

    out_path = os.path.join(PLOTS_DIR, "pss_correlation_peaks.png")
    visualize.plot_pss_correlation_peaks(
        result["corr_curves"], true_useful_offset,
        true_n_id_2, out_path
    )
    print(f"  Saved correlation-peak plot -> {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="SSB cell-search simulator")
    parser.add_argument("--trials", type=int, default=200,
                         help="Monte Carlo trials per SNR point (default: 200)")
    parser.add_argument("--snr", type=float, nargs="+",
                         default=[-14, -12, -10, -8, -6, -4, -2, 0, 2, 4],
                         help="List of SNR values in dB to sweep")
    parser.add_argument("--example-snr", type=float, default=-6,
                         help="SNR (dB) used for the single illustrative search")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) One illustrative single search + correlation-peak plot
    run_example_single_search(args.example_snr)

    # 2) Monte Carlo: detection probability vs SNR
    print("\nRunning Monte Carlo detection-vs-SNR sweep...")
    df = simulate.detection_vs_snr(args.snr, n_trials=args.trials, seed=args.seed)

    # 3) Cell-search time estimate
    df = simulate.estimate_cell_search_time(df)

    csv_path = os.path.join(RESULTS_DIR, "detection_vs_snr.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results table -> {csv_path}")

    visualize.plot_detection_vs_snr(df, os.path.join(PLOTS_DIR, "detection_vs_snr.png"))
    visualize.plot_search_time_vs_snr(df, os.path.join(PLOTS_DIR, "search_time_vs_snr.png"))
    print(f"Saved plots -> {PLOTS_DIR}/detection_vs_snr.png, "
          f"{PLOTS_DIR}/search_time_vs_snr.png")

    print("\nDone. See the results/ folder for all deliverables.")


if __name__ == "__main__":
    main()
