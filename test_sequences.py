"""
Basic sanity tests for PSS/SSS sequence generation and the end-to-end
cell-search pipeline. Run with:

    pytest -q
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssb_sim.pss import generate_pss
from ssb_sim.sss import generate_sss, cell_id
from ssb_sim.ssb import build_ssb_waveform, SYMBOL_LEN
from ssb_sim import channel
from ssb_sim.detector import full_cell_search
from ssb_sim import simulate


def test_pss_length_and_bpsk():
    for n in range(3):
        seq = generate_pss(n)
        assert len(seq) == 127
        assert np.allclose(np.abs(seq), 1.0)  # unit-modulus BPSK


def test_pss_orthogonality():
    """Different N_ID_2 PSS sequences must have low cross-correlation;
    a sequence must have perfect (normalised = 1) self-correlation."""
    seqs = [generate_pss(n) for n in range(3)]
    for i in range(3):
        self_corr = np.abs(np.vdot(seqs[i], seqs[i])) / 127
        assert np.isclose(self_corr, 1.0)
        for j in range(3):
            if i == j:
                continue
            cross_corr = np.abs(np.vdot(seqs[i], seqs[j])) / 127
            assert cross_corr < 0.1, "PSS sequences should be near-orthogonal"


def test_sss_length_and_bpsk():
    seq = generate_sss(n_id_1=17, n_id_2=1)
    assert len(seq) == 127
    assert np.allclose(np.abs(seq), 1.0)


def test_sss_unique_per_hypothesis():
    """Two different N_ID_1 values (same N_ID_2) must give different SSS
    sequences -- otherwise the UE could never resolve N_ID_1."""
    a = generate_sss(0, 0)
    b = generate_sss(1, 0)
    assert not np.allclose(a, b)


def test_cell_id_formula():
    assert cell_id(0, 0) == 0
    assert cell_id(1, 0) == 3
    assert cell_id(0, 1) == 1
    assert cell_id(335, 2) == 1007  # max physical cell ID in NR


def test_full_pipeline_high_snr_detects_correctly():
    """At a comfortably high SNR, the full cell search must succeed."""
    rng = np.random.default_rng(0)
    true_n_id_1, true_n_id_2 = 42, 2
    ssb = build_ssb_waveform(true_n_id_1, true_n_id_2)
    window_len = 4 * SYMBOL_LEN + simulate.SEARCH_WINDOW_MARGIN
    window, _ = channel.embed_in_search_window(ssb, window_len, rng)
    rx = channel.awgn(window, snr_db=15, rng=rng)

    result = full_cell_search(rx)
    assert result["n_id_2"] == true_n_id_2
    assert result["n_id_1"] == true_n_id_1
    assert result["n_id_cell"] == cell_id(true_n_id_1, true_n_id_2)


def test_detection_probability_improves_with_snr():
    """Sanity check the Monte Carlo driver itself on a tiny run: detection
    probability at a high SNR should be >= detection probability at a very
    low SNR."""
    df = simulate.detection_vs_snr([-20, 10], n_trials=20, seed=1)
    low_snr_p = df.loc[df.snr_db == -20, "detection_prob"].iloc[0]
    high_snr_p = df.loc[df.snr_db == 10, "detection_prob"].iloc[0]
    assert high_snr_p >= low_snr_p
