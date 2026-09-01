"""
ssb_sim -- a compact simulation of 5G NR SS/PBCH Block (SSB) generation
and UE-side initial cell search (PSS/SSS correlation-based detection).

Modules:
    pss.py         - PSS (Primary Synchronisation Signal) sequence generation
    sss.py         - SSS (Secondary Synchronisation Signal) sequence generation
    ofdm.py        - minimal OFDM mapping / IFFT-FFT helpers
    ssb.py         - assembles PSS + SSS + PBCH-placeholder into an SSB waveform
    channel.py     - AWGN + random timing-offset channel model
    detector.py    - UE-side PSS timing/ID search and SSS ID search
    simulate.py    - Monte Carlo driver (detection probability vs SNR, etc.)
    visualize.py   - plotting helpers
"""
