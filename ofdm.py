"""
Minimal OFDM helper used to turn frequency-domain PSS/SSS sequences
into a time-domain waveform, and back.

NOTE — SIMPLIFICATION FOR THIS SIMULATION:
This is a teaching/simulation model, not a full 5G NR PHY stack.
Real NR numerology (subcarrier spacing, exact FFT size, DC-subcarrier
puncturing, PBCH DM-RS, beam sweeping, multipath channel, etc.) is
simplified so the important concept -- PSS/SSS correlation-based cell
search -- is easy to follow and modify. This is called out explicitly
in the README.
"""

import numpy as np

N_FFT = 256          # FFT size used for the simulated OFDM symbol
CP_LEN = 18           # Cyclic prefix length (samples)
SEQ_LEN = 127          # PSS / SSS sequence length (fixed by 3GPP spec)


def map_sequence_to_symbol(seq: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    """
    Map a length-127 frequency-domain BPSK sequence onto the centre
    subcarriers of an OFDM symbol of size n_fft, and return the
    time-domain (IFFT) waveform.
    """
    assert len(seq) == SEQ_LEN
    grid = np.zeros(n_fft, dtype=np.complex128)
    half = SEQ_LEN // 2
    centre = n_fft // 2
    idx = np.arange(centre - half, centre + half + 1)
    grid[idx] = seq
    grid = np.fft.ifftshift(grid)
    time_symbol = np.fft.ifft(grid) * np.sqrt(n_fft)
    return time_symbol


def add_cp(time_symbol: np.ndarray, cp_len: int = CP_LEN) -> np.ndarray:
    """Prepend a cyclic prefix to a time-domain OFDM symbol."""
    return np.concatenate([time_symbol[-cp_len:], time_symbol])


def random_qpsk_symbol(n_fft: int = N_FFT, seed=None) -> np.ndarray:
    """
    Build a full-bandwidth random-QPSK OFDM symbol, used as a stand-in
    for the PBCH data that surrounds PSS/SSS inside a real SSB. It is
    NOT decoded anywhere -- it only contributes realistic "neighbour
    subcarrier" energy so the SSB waveform looks like a genuine burst.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_fft, 2))
    qpsk = (1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])
    qpsk /= np.sqrt(2)
    grid = np.fft.ifftshift(qpsk)
    time_symbol = np.fft.ifft(grid) * np.sqrt(n_fft)
    return time_symbol


def extract_symbol(rx_signal: np.ndarray, start: int, n_fft: int = N_FFT,
                    cp_len: int = CP_LEN) -> np.ndarray:
    """Slice out one OFDM symbol (skipping its CP) starting at `start`."""
    useful_start = start + cp_len
    return rx_signal[useful_start: useful_start + n_fft]


def symbol_to_freq(time_symbol: np.ndarray) -> np.ndarray:
    """FFT a received time-domain OFDM symbol back to the frequency grid."""
    n_fft = len(time_symbol)
    grid = np.fft.fft(time_symbol) / np.sqrt(n_fft)
    return np.fft.fftshift(grid)


def extract_centre_subcarriers(freq_grid: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Pull the centre `seq_len` subcarriers back out of a frequency grid."""
    n_fft = len(freq_grid)
    half = seq_len // 2
    centre = n_fft // 2
    idx = np.arange(centre - half, centre + half + 1)
    return freq_grid[idx]
