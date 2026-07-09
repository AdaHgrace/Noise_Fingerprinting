# Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models

A scalable pipeline for identifying quantum noise channels using classical shadow tomography and physics-informed feature engineering. Given measurement data from a fixed set of 3-qubit probe circuits, the pipeline classifies the underlying noise model among ten candidate channels using ensemble machine learning methods.

This repository accompanies the paper *"Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models."*

## Overview

Accurately identifying the dominant noise channel on a quantum device is a prerequisite for effective error mitigation, but full process tomography scales exponentially with system size. This project explores whether a lightweight, scalable alternative by combining randomized Pauli measurements, classical shadows with physics-informed feature engineering to reliably distinguish between common noise channels using only a small set of structured probe circuits.

The pipeline:
1. Prepares a fixed set of 3-qubit probe circuits: simple structured states + QAOA-style circuits
2. Executes them on a simulated noisy device: Qiskit Aer
3. Estimates Pauli observables via randomized classical shadow measurements
4. Builds a 279-dimensional feature vector per sample 
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

If you want to skip this step, use the pre-generated dataset located at:

```bash
data/qaoa_dataset.npz
```

```bash
python3 -m scripts.generate_dataset \
    --output data/my_dataset.npz \
    --samples-per-class 1000 \
    --shots 200 \
    --n-qubits 3 \
    --num-qaoa-probes 5 \
    --num-workers 4 \
    --noise-types all
```

This generates a labeled dataset and saves it as a compressed `.npz` file containing the feature matrix, labels, and per-sample metadata.

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--samples-per-class` | Number of samples per noise type | 500 |
| `--shots` | Classical shadow shots per probe circuit | 200 |
| `--n-qubits` | Number of qubits | 3 |
| `--num-qaoa-probes` | Number of QAOA-style probe circuits | 5 |
| `--noise-types` | Comma-separated list, or `all` | `all` |
| `--num-workers` | Parallel worker processes | 4 |

### 2. Train classifiers

```bash
python3 -m scripts.train_classifier \
    --dataset data/my_dataset.npz \
    --models extra_trees,random_forest,mlp
```

This trains the specified classifiers, evaluates them on a held-out test set, and saves trained models, confusion matrices, classification reports, and a `summary.json` with accuracy and macro F1 for each model to a timestamped folder under `results/`.

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--models` | Comma-separated: `random_forest`, `extra_trees`, `mlp` | all three |
| `--seed` | Random seed (affects model init) | 42 |
| `--dataset` | dataset to be used | files under data folder |
| `--output-dir` | Output directory | timestamped folder |

## Results

On a dataset of 14,000 labeled samples (1,400 per class), evaluated over three random seeds:

| Classifier | Accuracy (%) | Macro F1 |
|---|---|---|
| Random Forest | 84.26 ± 0.0036 | 0.8437 ± 0.0039 |
| Extra Trees | 84.06 ± 0.0019 | 0.8417 ± 0.0024 |
| MLP | 79.25 ± 0.0042 | 0.7924 ± 0.0046 |

See the paper for full confusion matrix analysis, scaling behavior, and discussion.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{jain2026shadow,
  title={Shadow-Based Noise Fingerprinting for Quantum Processors},
  author={Jain, Vridhi and Zhang, Lei},
  year={2026}
}
```

## License

MIT License.
