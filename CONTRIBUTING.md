# Contributing

This repository is a research release tied to a specific experimental protocol. Small fixes, portability improvements, tests, and documentation corrections are welcome.

Before opening a pull request:

1. run `python scripts/audit_public_release.py`;
2. run `python scripts/smoke_test.py`;
3. run the data-free tests listed in `README.md`;
4. describe any change to model behavior or numerical output explicitly;
5. do not commit imaging, patient-level predictions, Monte Carlo samples, logs, or submission materials.

Changes that alter a reported method should include a new configuration identifier and should not overwrite the frozen `configs/paper_model.json`, released fold assignment, checkpoint files, or checksum manifest.
