"""
Primary Synchronisation Signal (PSS) generation.

Reference: 3GPP TS 38.211, Section 7.4.2.2.1

The PSS is a length-127 BPSK sequence derived from a truncated
m-sequence. There are exactly 3 possible PSS sequences, corresponding
to N_ID_2 in {0, 1, 2} -- the physical-layer cell identity group
component that a UE can detect *without* knowing anything else about
the cell.
"""

import numpy as np

PSS_LENGTH = 127


def _pss_x_sequence():
    """
    Generate the length-127 base m-sequence x(i) used to build PSS,
    as defined in TS 38.211 7.4.2.2.1:

        x(i + 7) = (x(i + 4) + x(i)) mod 2,  i = 0 ... 119
        x(0..6)  = [0, 1, 1, 0, 1, 1, 1]
    """
    x = np.zeros(127, dtype=int)
    x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    return x


def generate_pss(n_id_2: int) -> np.ndarray:
    """
    Generate the frequency-domain PSS sequence (BPSK, length 127) for
    a given N_ID_2 in {0, 1, 2}.

    d_PSS(n) = 1 - 2 * x((n + 43 * N_ID_2) mod 127),  n = 0 ... 126
    """
    if n_id_2 not in (0, 1, 2):
        raise ValueError("N_ID_2 must be 0, 1, or 2")

    x = _pss_x_sequence()
    n = np.arange(PSS_LENGTH)
    m = (n + 43 * n_id_2) % PSS_LENGTH
    d_pss = 1 - 2 * x[m]
    return d_pss.astype(np.complex128)


def all_pss_sequences() -> dict:
    """Convenience helper: returns {n_id_2: pss_sequence} for all 3 roots."""
    return {k: generate_pss(k) for k in (0, 1, 2)}
