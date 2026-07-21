# BVI Navigation Simulation System

This repository contains the core simulation code for a study of non-visual outdoor navigation by blind and visually impaired (BVI) users. The model combines a navigation environment, risk inference, attention gating, and ACT-R based action selection to examine how environmental cues and cognitive load shape route-following behavior.

The current repository is intentionally limited to the system code. Calibration scripts, generated figures, cached map files, and previous simulation outputs are not included.

## Repository Structure

```text
BVI-SAS/        Core simulation modules
reports/        Empty output directory for generated reports
```

Main modules:

- `main.py`: command-line entry point for single-run and Monte Carlo simulations.
- `simulation.py`: simulation loop and ACT-R interaction logic.
- `environment.py`: map and route environment construction.
- `inference.py`: risk inference utilities.
- `actr_setup.py`: ACT-R model, buffers, chunks, and production setup.
- `reporting.py`: CSV, JSON, Markdown, and figure output generation.

## Requirements

The code was developed for Python 3.12 or later. Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy networkx osmnx pyactr matplotlib
```

## Running the Simulation

Run a single simulation from the repository root:

```bash
python BVI-SAS/main.py
```

Run with a familiarity condition:

```bash
python BVI-SAS/main.py --familiarity 0
python BVI-SAS/main.py --familiarity 1
```

Run multiple simulations:

```bash
python BVI-SAS/main.py --familiarity 1 --mc-runs 50 --seed-start 20260701
```

Generated files are written to `reports/`.

## Notes

The folder name `BVI-SAS` follows the project naming used for release. Because the hyphen is not a valid Python package character, the recommended entry point is direct script execution:

```bash
python BVI-SAS/main.py
```

## Citation

If this code is used in academic work, please cite the associated manuscript when available.

