"""
Secondary Synchronisation Signal (SSS) generation.

Reference: 3GPP TS 38.211, Section 7.4.2.3.1

The SSS is a length-127 BPSK sequence built from the product of two
shifted m-sequences. Once a UE has detected N_ID_2 from the PSS, it
correlates the received SSS symbol against all 336 possible N_ID_1
values (with the already-known N_ID_2) to resolve the full physical
cell identity:

    N_ID_cell = 3 * N_ID_1 + N_ID_2      (N_ID_1 in 0..335, N_ID_2 in 0..2)
"""

import numpy as np

SSS_LENGTH = 127


def _m_sequence(init_bits, feedback_taps):
    """
    Generic generator for the two length-127 m-sequences used by SSS.

    feedback_taps is a tuple (a, b) meaning:
        x(i + 7) = (x(i + a) + x(i + b)) mod 2
    """
    x = np.zeros(127, dtype=int)
    x[:7] = init_bits
    a, b = feedback_taps
    for i in range(120):
        x[i + 7] = (x[i + a] + x[i + b]) % 2
    return x


def _x0_sequence():
    # x0(i+7) = (x0(i+4) + x0(i)) mod 2 ; x0(0..6) = [1,0,0,0,0,0,0]
    return _m_sequence([1, 0, 0, 0, 0, 0, 0], (4, 0))


def _x1_sequence():
    # x1(i+7) = (x1(i+1) + x1(i)) mod 2 ; x1(0..6) = [1,0,0,0,0,0,0]
    return _m_sequence([1, 0, 0, 0, 0, 0, 0], (1, 0))


def generate_sss(n_id_1: int, n_id_2: int) -> np.ndarray:
    """
    Generate the frequency-domain SSS sequence (BPSK, length 127) for
    given N_ID_1 (0..335) and N_ID_2 (0..2).

    d_SSS(n) = [1 - 2*x0((n + m0) mod 127)] * [1 - 2*x1((n + m1) mod 127)]

    m0 = 15 * floor(N_ID_1 / 112) + 5 * N_ID_2
    m1 = N_ID_1 mod 112
    """
    if not (0 <= n_id_1 <= 335):
        raise ValueError("N_ID_1 must be in 0..335")
    if n_id_2 not in (0, 1, 2):
        raise ValueError("N_ID_2 must be 0, 1, or 2")

    x0 = _x0_sequence()
    x1 = _x1_sequence()

    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112

    n = np.arange(SSS_LENGTH)
    term0 = 1 - 2 * x0[(n + m0) % SSS_LENGTH]
    term1 = 1 - 2 * x1[(n + m1) % SSS_LENGTH]
    d_sss = term0 * term1
    return d_sss.astype(np.complex128)


def cell_id(n_id_1: int, n_id_2: int) -> int:
    """Combine N_ID_1 and N_ID_2 into the full physical cell ID (0..1007)."""
    return 3 * n_id_1 + n_id_2
