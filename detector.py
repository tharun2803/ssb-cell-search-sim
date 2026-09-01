"""
UE-side cell-search detector.

Stage 1 (PSS search):
    Slide the 3 candidate time-domain PSS waveforms (one per N_ID_2)
    across the received signal and find the sample offset + N_ID_2
    that gives the strongest correlation peak. This simultaneously
    solves symbol timing acquisition and N_ID_2 detection, exactly as
    a real UE's first cell-search step does.

Stage 2 (SSS search):
    Using the timing found in stage 1, jump forward to where the SSS
    OFDM symbol must be, FFT it, and correlate the centre subcarriers
    against all 336 candidate SSS sequences (for the already-known
    N_ID_2) to resolve N_ID_1.

    N_ID_cell = 3 * N_ID_1 + N_ID_2
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import ofdm
from .pss import generate_pss, all_pss_sequences
from .sss import generate_sss, cell_id
from .ssb import SYMBOL_LEN


def _time_domain_pss_replicas():
    """Pre-compute the 3 time-domain (no-CP) PSS waveforms once."""
    replicas = {}
    for n_id_2, seq in all_pss_sequences().items():
        replicas[n_id_2] = ofdm.map_sequence_to_symbol(seq)
    return replicas


_PSS_TIME_REPLICAS = _time_domain_pss_replicas()


def pss_search(rx_signal: np.ndarray):
    """
    Sliding correlation of the received signal against all 3 PSS
    time-domain replicas.

    Returns:
        best_offset   : sample index where PSS (useful part, post-CP)
                         best aligns
        best_n_id_2   : detected N_ID_2
        peak_value    : normalised correlation magnitude at the peak
        corr_curve    : dict {n_id_2: correlation-magnitude-vs-lag array}
                         (returned for plotting)
    """
    n_fft = ofdm.N_FFT
    n_samples = len(rx_signal)
    max_lag = n_samples - n_fft

    # All sliding windows of length n_fft, shape (max_lag+1, n_fft) -- a
    # vectorised replacement for an explicit per-lag Python loop.
    windows = sliding_window_view(rx_signal, n_fft)
    rx_energy = np.maximum(np.sum(np.abs(windows) ** 2, axis=1), 1e-12)

    corr_curves = {}
    best_offset, best_n_id_2, peak_value = 0, 0, -np.inf

    for n_id_2, replica in _PSS_TIME_REPLICAS.items():
        replica_energy = np.sum(np.abs(replica) ** 2)
        conj_replica = np.conj(replica)
        corr = np.abs(windows @ conj_replica)  # cross-correlation at every lag
        norm_corr = corr / np.sqrt(rx_energy * replica_energy)
        corr_curves[n_id_2] = norm_corr

        local_peak_idx = int(np.argmax(norm_corr))
        local_peak_val = norm_corr[local_peak_idx]
        if local_peak_val > peak_value:
            peak_value = local_peak_val
            best_offset = local_peak_idx
            best_n_id_2 = n_id_2

    return best_offset, best_n_id_2, peak_value, corr_curves


def sss_search(rx_signal: np.ndarray, pss_offset: int, n_id_2: int):
    """
    Given the PSS timing/offset and detected N_ID_2, locate the SSS
    OFDM symbol (2 symbols after PSS), FFT it, and correlate the
    centre subcarriers against all 336 candidate N_ID_1 hypotheses.

    Returns (best_n_id_1, best_correlation_score).
    """
    # SSS is symbol index 2 (0-based); PSS useful part started at pss_offset
    # so symbol 2's *CP* starts CP_LEN + 2 * SYMBOL_LEN samples later
    # relative to PSS symbol's own CP start. We know PSS's own CP started
    # CP_LEN samples before pss_offset.
    pss_cp_start = pss_offset - ofdm.CP_LEN
    sss_symbol_start = pss_cp_start + 2 * SYMBOL_LEN

    sss_time = ofdm.extract_symbol(rx_signal, sss_symbol_start)
    if len(sss_time) < ofdm.N_FFT:
        # Ran off the end of the buffer (can happen near noisy/incorrect peaks)
        return None, -np.inf

    freq_grid = ofdm.symbol_to_freq(sss_time)
    rx_sss = ofdm.extract_centre_subcarriers(freq_grid)

    ref_matrix, ref_norms = _sss_reference_bank(n_id_2)  # (336, 127), (336,)
    rx_norm = np.linalg.norm(rx_sss) + 1e-12
    scores = np.abs(ref_matrix.conj() @ rx_sss) / (ref_norms * rx_norm)

    best_n_id_1 = int(np.argmax(scores))
    best_score = float(scores[best_n_id_1])
    return best_n_id_1, best_score


_SSS_BANK_CACHE = {}


def _sss_reference_bank(n_id_2: int):
    """Pre-compute (and cache) all 336 candidate SSS sequences for a
    given N_ID_2, as a (336, 127) matrix, for fast batched correlation."""
    if n_id_2 not in _SSS_BANK_CACHE:
        bank = np.array([generate_sss(nid1, n_id_2) for nid1 in range(336)])
        norms = np.linalg.norm(bank, axis=1)
        _SSS_BANK_CACHE[n_id_2] = (bank, norms)
    return _SSS_BANK_CACHE[n_id_2]


def full_cell_search(rx_signal: np.ndarray):
    """
    Run the complete two-stage cell search and return a dict with all
    intermediate + final results.
    """
    pss_offset, n_id_2, pss_peak, corr_curves = pss_search(rx_signal)
    n_id_1, sss_score = sss_search(rx_signal, pss_offset, n_id_2)

    result = {
        "pss_offset": pss_offset,
        "n_id_2": n_id_2,
        "pss_peak": pss_peak,
        "n_id_1": n_id_1,
        "sss_score": sss_score,
        "n_id_cell": None if n_id_1 is None else cell_id(n_id_1, n_id_2),
        "corr_curves": corr_curves,
    }
    return result
