"""
smoke_test.py

Fast end-to-end smoke test for the noise fingerprinting pipeline.

Runs a tiny version of the full pipeline (few samples, few shots, all
n_qubits configurations) and prints intermediate values at each step,
so failures and successes are both visible and inspectable, not just
a pass/fail flag.

Usage:
    python3 smoke_test.py
"""

import sys
import traceback

import numpy as np


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_observables():
    print("\n[1/6] observables.generate_observables")
    print("-" * 60)
    from src.observables import generate_observables

    for n in [2, 3, 4]:
        observables, groups = generate_observables(n)
        expected_count = 3 * n * (n + 1) // 2

        print(f"  n_qubits={n}:")
        print(f"    total raw observables : {len(observables)} (expected {expected_count})")
        print(f"    x_like / y_like / z_like sizes : "
              f"{len(groups['x_like'])} / {len(groups['y_like'])} / {len(groups['z_like'])}")
        print(f"    example observables : {observables[:3]} ... {observables[-2:]}")

        check(
            len(observables) == expected_count,
            f"n={n}: expected {expected_count} observables, got {len(observables)}",
        )
        check(all(len(obs) == n for obs in observables), f"n={n}: observable length mismatch")
        check(len(groups["x_like"]) > 0, f"n={n}: x_like group empty")
        check(len(groups["y_like"]) > 0, f"n={n}: y_like group empty")
        check(len(groups["z_like"]) > 0, f"n={n}: z_like group empty")

    print("  PASS")


def test_circuits():
    print("\n[2/6] circuits.get_probe_circuits")
    print("-" * 60)
    from src.circuits import get_probe_circuits

    for n in [2, 3, 4]:
        probes = get_probe_circuits(
            n_qubits=n,
            num_qaoa_probes=5,
            seed=0,
            include_simple_probes=True,
        )
        probe_names = [name for name, _ in probes]
        depths = [circuit.depth() for _, circuit in probes]

        print(f"  n_qubits={n}:")
        print(f"    probes ({len(probes)}) : {probe_names}")
        print(f"    circuit depths        : {depths}")

        check(len(probes) == 6, f"n={n}: expected 6 probes, got {len(probes)}")
        for name, circuit in probes:
            check(circuit.num_qubits == n, f"n={n}: probe '{name}' has wrong qubit count")

    print("  PASS")


def test_noise_models():
    print("\n[3/6] noise_models.get_noise_model")
    print("-" * 60)
    from src.noise_models import get_noise_model, NOISE_TYPES

    for noise_type in NOISE_TYPES:
        model = get_noise_model(noise_type, strength=0.1)
        num_errors = len(model.to_dict().get("errors", []))
        print(f"  {noise_type:<25} -> built OK ({num_errors} quantum error entries)")
        check(model is not None, f"noise model '{noise_type}' returned None")

    print(f"  PASS ({len(NOISE_TYPES)} noise types)")


def test_shadow():
    print("\n[4/6] shadow.run_shadow_and_estimate")
    print("-" * 60)
    from src.circuits import build_plus_circuit
    from src.observables import generate_observables
    from src.shadow import run_shadow_and_estimate

    n = 2
    circuit = build_plus_circuit(n)
    observables, _ = generate_observables(n)

    values, shadow_data = run_shadow_and_estimate(
        circuit=circuit,
        observables=observables,
        noise_model=None,
        shots=20,
        seed=0,
    )

    print(f"  n_qubits={n}, shots=20, noise=None (ideal circuit)")
    print(f"    observables       : {observables}")
    print(f"    estimated values  : {[round(v, 3) for v in values]}")
    print(f"    value range       : [{values.min():.3f}, {values.max():.3f}]")
    print(f"    unique bases seen : {len(set(shadow_data['basis_strings']))} / 9 possible")

    check(len(values) == len(observables), "shadow output length mismatch")
    check(np.all(np.abs(values) <= 1.0 + 1e-6), "shadow values out of [-1, 1] range")

    print("  PASS")


def test_fingerprint():
    print("\n[5/6] fingerprint.build_fingerprint")
    print("-" * 60)
    from src.fingerprint import build_fingerprint

    for n in [2, 3, 4]:
        vec = build_fingerprint(
            noise_type="depolarizing",
            strength=0.1,
            n_qubits=n,
            shots=20,
            seed=0,
            num_qaoa_probes=2,
            include_simple_probes=True,
            include_derived_features=True,
        )
        num_probes = 6
        raw_per_probe = 3 * n * (n + 1) // 2
        expected_dim = num_probes * (raw_per_probe + 10)

        print(f"  n_qubits={n}:")
        print(f"    fingerprint dim  : {vec.shape[0]} (expected {expected_dim})")
        print(f"    value range      : [{vec.min():.3f}, {vec.max():.3f}]")
        print(f"    first 5 values   : {[round(v, 3) for v in vec[:5]]}")
        print(f"    any NaN/Inf      : {not np.all(np.isfinite(vec))}")

        check(
            vec.shape[0] == expected_dim,
            f"n={n}: expected fingerprint dim {expected_dim}, got {vec.shape[0]}",
        )
        check(np.all(np.isfinite(vec)), f"n={n}: fingerprint contains non-finite values")

    print("  PASS")


def test_end_to_end_training():
    print("\n[6/6] tiny end-to-end dataset + classifier training")
    print("-" * 60)
    from src.fingerprint import build_fingerprint
    from src.noise_models import NOISE_TYPES
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import ExtraTreesClassifier

    n_qubits = 2
    samples_per_class = 5

    X, y = [], []
    for class_id, noise_type in enumerate(NOISE_TYPES):
        for i in range(samples_per_class):
            vec = build_fingerprint(
                noise_type=noise_type,
                strength=0.1,
                n_qubits=n_qubits,
                shots=20,
                seed=class_id * 100 + i,
                num_qaoa_probes=2,
                include_simple_probes=True,
                include_derived_features=True,
            )
            X.append(vec)
            y.append(class_id)

    X = np.array(X)
    y = np.array(y)

    print(f"  dataset shape : X={X.shape}, y={y.shape}")
    print(f"  classes       : {len(NOISE_TYPES)} ({samples_per_class} samples each)")

    check(X.shape[0] == len(NOISE_TYPES) * samples_per_class, "dataset size mismatch")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y,
    )
    print(f"  split         : train={X_train.shape[0]}, test={X_test.shape[0]}")

    clf = ExtraTreesClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    preds = clf.predict(X_test)

    print(f"  train accuracy : {train_acc:.3f}")
    print(f"  test accuracy  : {test_acc:.3f}  (not meaningful at this scale, pipeline check only)")
    print(f"  test preds     : {preds.tolist()}")
    print(f"  test labels    : {y_test.tolist()}")

    check(0.0 <= test_acc <= 1.0, "invalid accuracy returned")

    print("  PASS")


def main():
    print("=" * 60)
    print("SMOKE TEST: noise fingerprinting pipeline")
    print("=" * 60)

    tests = [
        test_observables,
        test_circuits,
        test_noise_models,
        test_shadow,
        test_fingerprint,
        test_end_to_end_training,
    ]

    failed = []

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            failed.append((test_fn.__name__, e))
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failed:
        print(f"RESULT: {len(failed)}/{len(tests)} test(s) FAILED")
        for name, e in failed:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print(f"RESULT: {len(tests)}/{len(tests)} test(s) PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()