# Consistency Model Code

`cm_purifier/` contains the new pixel-space poison-aware Consistency Model
implementation for Algorithm 2 in the project README.

The `InstantPure/` directory is kept as a reference implementation of the CVPR
2025 method, but the new purifier code does not import from it.

The final bank has 40,000 matched pairs: 10,000 clean identity, 10,000 WB, 10,000 BP, and 10,000 NS. Narcissus filenames use `ns_c<semantic>_t<trigger-target>_<index>.png`; the loader preserves the semantic label and assigns attack ID 3. Training accepts `--gamma-ns` (runner variable `GAMMA_NS`), default 1.0. The dataset smoke check therefore expects 40,000 total pairs and 4,000 samples per class.
