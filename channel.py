"""
Simple channel model: places the SSB waveform at a random, unknown
sample offset inside a longer noise-only search window, and adds
complex AWGN at a specified SNR (defined per-sample, referenced to
the SSB waveform's average power).
"""

import numpy as np


def awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN to `signal` so that average SNR = snr_db (dB)."""
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


def embed_in_search_window(ssb_waveform: np.ndarray, window_len: int,
                            rng: np.random.Generator):
    """
    Place `ssb_waveform` at a random start offset inside a zero
    (pure-noise-to-be-added) window of length `window_len`, mimicking
    the unknown symbol timing a UE faces at cell search. Returns
    (window, true_offset).
    """
    assert window_len > len(ssb_waveform)
    max_offset = window_len - len(ssb_waveform)
    true_offset = rng.integers(0, max_offset)
    window = np.zeros(window_len, dtype=np.complex128)
    window[true_offset: true_offset + len(ssb_waveform)] = ssb_waveform
    return window, true_offset
