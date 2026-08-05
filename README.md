# Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models

A scalable pipeline for identifying quantum noise channels using classical shadow tomography and physics-informed feature engineering. Given measurement data from a fixed set of probe circuits, the pipeline classifies the underlying noise model among ten candidate channels using ensemble machine learning methods.

This repository accompanies the paper *"Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models."*

## Overview

Accurately identifying the dominant noise channel on a quantum device is a prerequisite for effective error mitigation, but full process tomography scales exponentially with system size. This project explores whether a lightweight, scalable alternative by combining randomized Pauli measurements, classical shadows with physics-informed feature engineering to reliably distinguish between common noise channels using only a small set of structured probe circuits.

The pipeline:
1. Prepares a fixed set of probe circuits at a given qubit count: simple structured states (basis states, superposition, GHZ) + QAOA-style circuits
2. Executes them on a simulated noisy device (Qiskit Aer)
3. Estimates Pauli observables via randomized classical shadow measurements
4. Builds a feature vector per sample, scaling with qubit count (252-dimensional at 3 qubits)
5. Classifies the noise type using Random Forest, Extra Trees, or an MLP

## Noise types covered

Depolarizing, amplitude damping, phase damping, phase-amplitude damping, thermal relaxation, bit flip, phase flip, Pauli-asymmetric, readout error, and reset.

## Repository structure

```
Noise_Fingerprinting/
├── src/
│   ├── noise_models.py      # Qiskit Aer noise model definitions
│   ├── circuits.py           # Probe circuit construction
│   ├── observables.py        # Pauli observable set and groupings
│   ├── shadow.py              # Classical shadow tomography
│   └── fingerprint.py        # Feature vector construction
├── scripts/
│    ├── vizualisation/
│      ├── plot_confusion_matrix.py # plotting confusion matrix
│      ├── plot_qubit_analysis.py # scaling with number of qubit
│      ├── plot_scaling.py # scaling with dataset size  
│      └── plot_strength_analysis.py  # Noise strength
│   ├── generate_dataset.py  # Build a labeled dataset
│   └── train_classifier.py  # Train and evaluate classifiers
├── data/                     # Generated datasets (.npz)
├── results/                  # Training outputs, metrics, models
├── requirements.txt              
└── smoke_test.py

```

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/AdaHgrace/Noise_Fingerprinting.git
cd Noise_Fingerprinting

python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

**Core dependencies:** `qiskit`, `qiskit-aer`, `numpy`, `scikit-learn`, `joblib`, `tqdm`, `matplotlib`, `seaborn`

## Usage

### 0. Smoke Test

Run a fast smoke test to check that the pipeline works:

```bash
python3 -m smoke_test
```
### 1. Generate a dataset

If you want to skip this step, pre-generated datasets are provided under `data/`, covering all qubit counts and strength ranges used in the paper:

| File | Qubits | Strength range | Role |
|---|---|---|---|
| `data/qaoa_dataset_n3_0.1_0.5.npz` | 3 | $[0.1, 0.5]$ | Primary experimental condition (main results, Table 1) |
| `data/qaoa_dataset_n3_0.05_0.25.npz` | 3 | $[0.05, 0.25]$ | Strength-range sweep |
| `data/qaoa_dataset_n3_0.1_1.0.npz` | 3 | $[0.1, 1.0]$ | Strength-range sweep |
| `data/qaoa_dataset_n2_0.1_0.5.npz` | 2 | $[0.1, 0.5]$ | Qubit-count sweep |
| `data/qaoa_dataset_n4_0.1_0.5.npz` | 4 | $[0.1, 0.5]$ | Qubit-count sweep |

Each dataset contains 10,000 labeled samples (1,000 per noise model). To reproduce the main results table, use `qaoa_dataset_n3_0.1_0.5.npz`.

```bash
python3 -m scripts.generate_dataset --output data/my_dataset.npz --samples-per-class 1000 --shots 200 --n-qubits 3 --num-qaoa-probes 5 --num-workers 4 --noise-types all --min-strength 0.1 --max-strength 0.5
```

This generates a labeled dataset and saves it as a compressed `.npz` file containing the feature matrix, labels, and per-sample metadata.

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--samples-per-class` | Number of samples per noise type | 1000 |
| `--shots` | Classical shadow shots per probe circuit | 200 |
| `--n-qubits` | Number of qubits | 3 |
| `--num-qaoa-probes` | Number of QAOA-style probe circuits | 5 |
| `--noise-types` | Comma-separated list, or `all` | `all` |
| `--num-workers` | Parallel worker processes | 4 |
| `--min-strength` | Minimum noise strength| 0.05 |
| `--max-strength` | Maximum noise strength | 0.25 |
| `--checkpoint-every` | Save a resumable checkpoint every *N* successful samples | 1000 |
| `--resume` | Resume from the latest compatible checkpoint for the specified output dataset | Disabled |

### Resuming an interrupted run

Dataset generation automatically saves checkpoints every `--checkpoint-every`
successful samples.

If a run is interrupted, resume it by
rerunning the same command with the `--resume` flag:

```bash
python3 -m scripts.generate_dataset --output data/my_dataset.npz --samples-per-class 1000 --shots 200 --n-qubits 3 --num-qaoa-probes 5 --num-workers 4 --noise-types all --min-strength 0.1 --max-strength 0.5 --resume
```

The script will automatically:

- locate the latest checkpoint for the specified output dataset,
- restore all previously generated samples,
- generate only the remaining samples, and
- write the completed dataset to the requested output file.

**Note:** Resume only works when all dataset-generation parameters match the
original run (number of qubits, strength range, probe configuration, noise
types, etc.). If they differ, the script will report the mismatch rather than
mix incompatible samples into the same dataset.

### 2. Train classifiers

```bash
python3 -m scripts.train_classifier --dataset data/my_dataset.npz --models extra_trees,random_forest,mlp --seed 44
```

This trains the specified classifiers, evaluates them on a held-out test set, and saves trained models, confusion matrices, classification reports, and a `summary.json` with accuracy and macro F1 for each model to a timestamped folder under `results/`.

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--models` | Comma-separated: `random_forest`, `extra_trees`, `mlp` | all three |
| `--seed` | Random seed (affects model init) | 44 |
| `--dataset` | dataset to be used | files under data folder |
| `--output-dir` | Output directory | timestamped folder |

## Results

On our primary experimental condition (3 qubits, noise strength sampled from $[0.1, 0.5]$, 10,000 labeled samples, 1,000 per class), evaluated over three random seeds (42, 43, 44):

| Classifier | Accuracy | Macro F1 |
|---|---|---|
| Extra Trees | 0.7358 ± 0.0064 | 0.7288 ± 0.0064 |
| Random Forest | 0.7355 ± 0.0104 | 0.7299 ± 0.0113 |
| MLP | 0.7150 ± 0.0115 | 0.6987 ± 0.0237 |

Extra trees and random forest achieve statistically indistinguishable accuracy, both outperforming the MLP by 2–3 percentage points on both metrics.

### Effect of noise strength

Test accuracy at 3 qubits across three noise-strength sampling ranges, mean ± std over three seeds:

| Strength range | Extra Trees | Random Forest | MLP |
|---|---|---|---|
| $[0.05, 0.25]$ | 0.7078 ± 0.0070 | 0.6990 ± 0.0106 | 0.7025 ± 0.0004 |
| $[0.1, 0.5]$   | 0.7358 ± 0.0064 | 0.7355 ± 0.0104 | 0.7150 ± 0.0115 |
| $[0.1, 1.0]$   | 0.6545 ± 0.0055 | 0.6508 ± 0.0046 | 0.6248 ± 0.0018 |

Accuracy peaks at the intermediate range $[0.1, 0.5]$ and is lower at both the narrower low-strength range and the widest range tested, consistently across all three classifiers.

### Effect of qubit count

Test accuracy at the $[0.1, 0.5]$ strength range across qubit counts, mean ± std over three seeds:

| Qubit count | Extra Trees | Random Forest | MLP |
|---|---|---|---|
| 2 | 0.7467 ± 0.0063 | 0.7527 ± 0.0055 | 0.7327 ± 0.0088 |
| 3 | 0.7358 ± 0.0064 | 0.7355 ± 0.0104 | 0.7150 ± 0.0115 |
| 4 | 0.7340 ± 0.0054 | 0.7445 ± 0.0053 | 0.7140 ± 0.0151 |

Classification accuracy is largely insensitive to qubit count over the range tested (2 to 4 qubits), in contrast to its strong, non-monotonic dependence on the noise-strength sampling range. See the paper for full confusion matrix analysis and discussion.


## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{jain2026shadow,
  title     = {Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models},
  author    = {Jain, Vridhi and Zhang, Lei},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Quantum Computing and Engineering (QCE) Workshops},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

MIT License.
