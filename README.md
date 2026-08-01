# pHTag code

This folder contains the pH model used in the paper and four small examples.

Install the three dependencies:

```bash
pip install numpy pandas scipy
```

Run one file:

```bash
python predict.py examples/static_good.csv.gz
```

Run several files together:

```bash
python predict.py examples/static_good.csv.gz examples/static_hard.csv.gz examples/motion_good.csv.gz examples/motion_hard.csv.gz
```

Expected output:

```text
session_id,predicted_ph
MOTION_GOOD,9.503586
MOTION_HARD,9.000773
STATIC_GOOD,9.872972
STATIC_HARD,9.394300
```

The input files contain scalar reader phase represented as sine/cosine and
RSSI. The code builds the dry-baseline dual-tag D4 features, averages the eight
saved Ridge outputs, and clips the result to pH 7--10.

## HFSS model

`hfss/u8_sot23_ring_pedot.aedt` is the nominal on-body antenna model used as
the reference geometry for the Chapter 4 HFSS sweeps. It was created with
Ansys Electronics Desktop 2025 R1 and includes the simplified layered foot
loading used in the paper.

The HFSS project is not used by `predict.py`, does not generate the example
reader CSV files, and is not part of the saved Ridge pH prediction model.
