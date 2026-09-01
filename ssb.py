"""
Build a simplified 4-OFDM-symbol SS/PBCH Block (SSB) time-domain waveform:

    Symbol 0 : PSS
    Symbol 1 : PBCH placeholder (random QPSK, not decoded)
    Symbol 2 : SSS (centre subcarriers) + PBCH placeholder elsewhere
    Symbol 3 : PBCH placeholder (random QPSK, not decoded)

This mirrors the real NR SSB symbol ordering (PSS, PBCH, SSS+PBCH,
PBCH) closely enough to demonstrate the cell-search timing problem,
without implementing the full PBCH payload/DM-RS chain.
"""

import numpy as np
from . import ofdm
from .pss import generate_pss
from .sss import generate_sss

SYMBOLS_PER_SSB = 4
SYMBOL_LEN = ofdm.N_FFT + ofdm.CP_LEN  # samples per OFDM symbol incl. CP


def build_ssb_waveform(n_id_1: int, n_id_2: int, seed=None) -> np.ndarray:
    """Return the full time-domain SSB waveform (4 OFDM symbols, with CP)."""
    pss_seq = generate_pss(n_id_2)
    sss_seq = generate_sss(n_id_1, n_id_2)

    # Symbol 0: PSS
    sym0 = ofdm.add_cp(ofdm.map_sequence_to_symbol(pss_seq))

    # Symbol 1: PBCH placeholder
    sym1 = ofdm.add_cp(ofdm.random_qpsk_symbol(seed=seed))

    # Symbol 2: SSS on centre subcarriers, PBCH placeholder elsewhere
    sss_time = ofdm.map_sequence_to_symbol(sss_seq)
    pbch_fill = ofdm.random_qpsk_symbol(seed=None if seed is None else seed + 1)
    # Combine in frequency domain instead of naive time-domain addition
    grid_sss = np.fft.ifftshift(np.zeros(ofdm.N_FFT, dtype=np.complex128))
    half = 127 // 2
    centre = ofdm.N_FFT // 2
    idx = np.arange(centre - half, centre + half + 1)
    fill_grid = np.fft.fftshift(np.fft.fft(pbch_fill) / np.sqrt(ofdm.N_FFT))
    fill_grid[idx] = sss_seq
    combined = np.fft.ifftshift(fill_grid)
    sym2_time = np.fft.ifft(combined) * np.sqrt(ofdm.N_FFT)
    sym2 = ofdm.add_cp(sym2_time)

    # Symbol 3: PBCH placeholder
    sym3 = ofdm.add_cp(ofdm.random_qpsk_symbol(seed=None if seed is None else seed + 2))

    waveform = np.concatenate([sym0, sym1, sym2, sym3])
    return waveform
