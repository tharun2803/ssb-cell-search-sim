# Project Report: SS/PBCH Block (SSB) and Initial Cell-Search Timing Simulator

## Abstract

Initial cell search is the first procedure a 5G NR User Equipment (UE)
performs on power-up: without any prior knowledge of timing, frequency, or
cell identity, it must locate a broadcast synchronisation signal and derive
the serving cell's Physical Cell ID. This project simulates that process in
Python, using the exact 3GPP-specified Primary and Secondary
Synchronisation Signal (PSS/SSS) sequences, a simplified OFDM/AWGN channel
model, and a two-stage correlation-based detector. The simulator measures
detection reliability as a function of SNR, visualises the correlation
peaks used for timing acquisition, and estimates the expected cell-search
time a UE would experience under different channel conditions.

---

## 1. Problem Statement

A UE finds a cell by detecting the SS/PBCH Block (SSB) — specifically, by
correlating received samples against known PSS and SSS reference sequences
(PSS/SSS correlation) and subsequently decoding the PBCH. This project
models the cell-search procedure that runs at the start of every network
"attach," from raw synchronisation-signal generation through to correctly
identifying a simulated cell's Physical Cell ID under noise.

## 2. Objectives

1. Simulate SSB structure in Python.
2. Implement cell-ID detection via PSS/SSS correlation.
3. Measure detection reliability vs. SNR.
4. Estimate cell-search time.
5. Visualise the correlation peaks.

Each objective is addressed directly by a corresponding module/output, as
summarised in the table below.

| Objective | Implementation | Deliverable |
|---|---|---|
| Simulate SSB structure | `ssb_sim/pss.py`, `sss.py`, `ofdm.py`, `ssb.py` | 4-symbol SSB waveform (PSS / PBCH / SSS+PBCH / PBCH) |
| Cell-ID detection via correlation | `ssb_sim/detector.py` | Two-stage PSS timing search + SSS ID search |
| Detection reliability vs. SNR | `ssb_sim/simulate.py` (Monte Carlo) | `results/detection_vs_snr.csv`, `detection_vs_snr.png` |
| Cell-search time estimate | `simulate.estimate_cell_search_time()` | `search_time_vs_snr.png` |
| Correlation peak visualisation | `ssb_sim/visualize.py` | `pss_correlation_peaks.png` |

---

## 3. System Model

### 3.1 Synchronisation signal generation

Both PSS and SSS are generated exactly as specified in **3GPP TS 38.211
§7.4.2**:

- **PSS**: a length-127 BPSK sequence derived from a truncated m-sequence,
  parameterised by `N_ID_2 ∈ {0, 1, 2}`. There are only 3 possible PSS
  sequences, and they are close to mutually orthogonal (see §5.1), which is
  what allows a UE to detect the correct one reliably even before any other
  cell information is known.

- **SSS**: a length-127 BPSK sequence formed from the product of two
  shifted m-sequences, parameterised by `N_ID_1 ∈ {0, ..., 335}` and the
  already-known `N_ID_2`. There are 336 possible SSS sequences per `N_ID_2`.

Combining both parameters gives the full Physical Cell ID:

```
N_ID_cell = 3 × N_ID_1 + N_ID_2        (range: 0 to 1007)
```

### 3.2 SSB waveform construction

The simulated SSB is a 4-symbol OFDM waveform:

```
Symbol 0: PSS
Symbol 1: PBCH placeholder (random QPSK, not decoded)
Symbol 2: SSS (centre subcarriers) + PBCH placeholder (remaining subcarriers)
Symbol 3: PBCH placeholder (random QPSK, not decoded)
```

This mirrors the real NR SSB symbol ordering closely enough to demonstrate
the cell-search timing problem realistically, while keeping the
implementation compact (see §6 for simplifications relative to the full
specification).

### 3.3 Channel model

The SSB waveform is placed at a **random, unknown sample offset** inside a
longer window (simulating the fact that a UE has no a-priori symbol timing
at power-up), and complex AWGN is added at a controlled per-waveform SNR.

### 3.4 Receiver (UE-side cell search)

**Stage 1 — PSS search (timing + N_ID_2):**
All three candidate time-domain PSS waveforms are slid across the received
signal. A normalised cross-correlation is computed at every lag for every
candidate; the lag/candidate pair giving the strongest peak is taken as the
detected symbol timing and `N_ID_2`. This single step solves both timing
acquisition and partial cell identification simultaneously — exactly as in
a real UE baseband.

**Stage 2 — SSS search (N_ID_1):**
Using the timing found in Stage 1, the receiver locates the SSS-bearing
OFDM symbol (2 symbols after PSS), FFTs it, extracts the centre
subcarriers, and correlates against all 336 candidate SSS sequences for the
already-known `N_ID_2`. The best-scoring `N_ID_1` is selected.

The two results are then combined into `N_ID_cell`.

---

## 4. Methodology

- **Sequence validation**: Before running any noisy simulation, the PSS/SSS
  generators are unit-tested for correct length, unit-modulus (BPSK)
  property, and near-orthogonality between distinct sequences (see
  `tests/test_sequences.py`).
- **Single-trial illustration**: One example cell search (arbitrary cell:
  `N_ID_1=200, N_ID_2=1` → `N_ID_cell=601`) is run at a representative
  negative SNR to produce the correlation-peak visualisation.
- **Monte Carlo sweep**: For each SNR point in a swept range, `N` trials
  (default 150–200) are run with a **randomly chosen cell identity per
  trial**, and the fraction of trials in which the full `N_ID_cell` is
  correctly recovered is recorded as the empirical detection probability.
  The fraction where `N_ID_2` alone is correctly recovered (Stage 1 only)
  is also recorded, to separate PSS-stage performance from full end-to-end
  performance.
- **Cell-search time estimate**: Modelled as a simple geometric-expectation
  process — if a single SSB occasion is detected with empirical probability
  `p`, the expected number of occasions needed is `1/p`, giving:

  ```
  E[search time] = (1 / p) × SSB_period_ms + processing_time_ms
  ```

  using a default SSB period of 20 ms and 2 ms of processing overhead.

---

## 5. Results

### 5.1 Sequence properties (sanity checks)

All PSS self-correlations are exactly 1.0 (normalised), and all
cross-correlations between distinct `N_ID_2` sequences are below 0.01 —
confirming the sequences are effectively orthogonal, which is essential for
reliable `N_ID_2` disambiguation.

### 5.2 Correlation peak visualisation

`results/plots/pss_correlation_peaks.png` shows the normalised correlation
magnitude vs. lag for all 3 candidate PSS sequences at SNR = −8 dB. The
correct `N_ID_2` sequence produces a distinct spike exactly at the true
symbol boundary, clearly separated from the noise floor, while incorrect
candidates remain within the noise floor across all lags — visually
demonstrating why sliding PSS correlation is such a robust timing
acquisition method even in low-SNR conditions.

### 5.3 Detection reliability vs. SNR

A full waterfall curve was obtained by sweeping SNR from −20 dB to 0 dB
(200 trials per point, random cell identity each trial):

| SNR (dB) | P(N_ID_2 correct) | P(full N_ID_cell correct) |
|---:|---:|---:|
| −20 | 0.305 | 0.000 |
| −18 | 0.345 | 0.015 |
| −16 | 0.455 | 0.060 |
| −14 | 0.605 | 0.255 |
| −12 | 0.830 | 0.725 |
| −10 | 0.990 | 0.970 |
| −8  | 1.000 | 1.000 |
| −6 to 0 | 1.000 | 1.000 |

*(Full data: `results/detection_vs_snr.csv`)*

Two effects are visible:
1. **Stage 1 (PSS/N_ID_2) is consistently easier than the full chain** —
   it only has to distinguish among 3 candidates, whereas Stage 2 (SSS)
   must distinguish among 336. At low SNR, `N_ID_2` is often detected
   correctly even when the full cell ID is not.
2. **A sharp transition region** exists between roughly −16 dB and −10 dB,
   typical of correlation-based detection: below this range performance
   collapses toward chance level, above it performance saturates near
   100%.

### 5.4 Estimated cell-search time

Combining the measured detection probabilities with a 20 ms SSB period
gives an expected search time that stays close to the theoretical floor
(~22 ms — one SSB period plus processing time) once SNR is ≥ −8 dB, but
grows sharply below −12 dB as the UE must wait through many repeated SSB
occasions before a detection succeeds. Below −18 dB the expected wait
becomes impractically long, illustrating why real UEs rely on additional
techniques (non-coherent combining across multiple SSB occasions, wider
search windows, etc.) in genuinely poor coverage.

---

## 6. Limitations

This project is a **simulation for demonstrating the cell-search concept**,
not a full 5G NR PHY implementation:

- OFDM numerology (FFT size, CP length, subcarrier spacing) is simplified and not tied to a specific real NR configuration.
- The DC subcarrier is not punctured as in the real specification.
- PBCH is represented only as filler QPSK energy; the MIB payload, PBCH DM-RS, and Polar coding are not implemented.
- Only AWGN is modelled — no multipath fading, carrier-frequency offset, or beam sweeping across a full SSB burst set.
- The cell-search-time model is a simplified geometric-expectation estimate, not a full RRC/NAS attach-procedure timing analysis.

## 7. Conclusion

The simulator successfully reproduces the core mechanics of 5G NR initial
cell search: generating spec-accurate PSS/SSS sequences, constructing an
SSB waveform, and recovering the full Physical Cell ID through two-stage
correlation under AWGN. The resulting detection-vs-SNR waterfall curve and
correlation-peak visualisations match the expected qualitative behaviour of
a real correlation-based synchronisation receiver, providing a compact,
extensible foundation for further exploration (e.g. adding CFO, multipath,
or full PBCH decoding).
