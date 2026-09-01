"""
Monte Carlo driver: runs the full cell-search chain many times across
a range of SNRs and measures how often the UE correctly recovers
(N_ID_1, N_ID_2) -> N_ID_cell, plus a simple cell-search-time estimate
built from the measured detection probability.
"""

import numpy as np
import pandas as pd

from . import channel
from .ssb import build_ssb_waveform, SYMBOL_LEN
from .detector import full_cell_search
from .sss import cell_id

SEARCH_WINDOW_MARGIN = 400  # extra "unknown timing" samples around the SSB


def run_single_trial(true_n_id_1: int, true_n_id_2: int, snr_db: float,
                      rng: np.random.Generator):
    """Build one noisy SSB, run cell search, and report success/failure."""
    ssb_wave = build_ssb_waveform(true_n_id_1, true_n_id_2)
    window_len = 4 * SYMBOL_LEN + SEARCH_WINDOW_MARGIN
    window, true_offset = channel.embed_in_search_window(ssb_wave, window_len, rng)
    rx = channel.awgn(window, snr_db, rng)

    result = full_cell_search(rx)
    true_cell_id = cell_id(true_n_id_1, true_n_id_2)

    success = (result["n_id_cell"] == true_cell_id)
    return success, result


def detection_vs_snr(snr_range_db, n_trials=200, seed=42):
    """
    For each SNR in snr_range_db, run n_trials random-cell trials and
    record the empirical detection probability (fraction of trials
    where the full N_ID_cell was recovered correctly).

    Returns a pandas DataFrame with columns: snr_db, detection_prob,
    n_trials, pss_only_detection_prob (N_ID_2 correct, ignoring SSS).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for snr_db in snr_range_db:
        n_success = 0
        n_id2_success = 0
        for _ in range(n_trials):
            true_n_id_1 = int(rng.integers(0, 336))
            true_n_id_2 = int(rng.integers(0, 3))
            success, result = run_single_trial(true_n_id_1, true_n_id_2, snr_db, rng)
            n_success += int(success)
            n_id2_success += int(result["n_id_2"] == true_n_id_2)

        rows.append({
            "snr_db": snr_db,
            "detection_prob": n_success / n_trials,
            "pss_only_detection_prob": n_id2_success / n_trials,
            "n_trials": n_trials,
        })
        print(f"  SNR = {snr_db:5.1f} dB  ->  "
              f"P(full cell-ID correct) = {n_success/n_trials:5.3f}   "
              f"P(N_ID_2 correct) = {n_id2_success/n_trials:5.3f}")

    return pd.DataFrame(rows)


def estimate_cell_search_time(df: pd.DataFrame, ssb_period_ms: float = 20.0,
                               processing_time_ms: float = 2.0):
    """
    Very simple cell-search time estimator.

    A UE that fails to detect the SSB on one occasion simply waits for
    the next periodic SSB burst and tries again. If detection on a
    single SSB occasion succeeds with probability p (measured via
    Monte Carlo at a given SNR), the number of SSB occasions needed is
    geometrically distributed with mean 1/p, so:

        E[cell-search time] = (1 / p) * ssb_period_ms + processing_time_ms

    This is a standard back-of-the-envelope model used to reason about
    cell-search latency in poor coverage; it ignores beam-sweeping,
    multiple SSBs per burst set, and higher-layer signalling delay.
    """
    out = df.copy()
    p = out["detection_prob"].clip(lower=1e-4)  # avoid division blow-up at 0
    out["expected_search_time_ms"] = (1.0 / p) * ssb_period_ms + processing_time_ms
    return out
