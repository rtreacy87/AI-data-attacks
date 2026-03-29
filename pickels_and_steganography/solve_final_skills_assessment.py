import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.multiclass import OneVsRestClassifier


DEFAULT_DATASET = "assessment_dataset.npz"
DEFAULT_BASE_URL = "http://154.57.164.83:32387"
DEFAULT_EVALUATE_PATH = "/evaluate_model"
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_SEED = 1337


# Candidate attack settings:
# (fraction of class-1 points to flip, fraction of flipped points sent to class 0)
# Remaining flipped points are sent to class 2.
CANDIDATE_CONFIGS = [
    (0.90, 0.25),
    (0.90, 0.30),
    (0.90, 0.35),
    (0.85, 0.25),
    (0.85, 0.30),
    (0.85, 0.35),
    (0.95, 0.25),
    (0.95, 0.30),
    (0.95, 0.35),
    (0.90, 0.65),
    (0.85, 0.65),
    (0.95, 0.65),
]


def load_dataset(path):
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def poison_labels(y_train, flip_fraction, ratio_to_class0, seed):
    if not 0 <= flip_fraction <= 1:
        raise ValueError("flip_fraction must be in [0, 1].")
    if not 0 <= ratio_to_class0 <= 1:
        raise ValueError("ratio_to_class0 must be in [0, 1].")

    y_poisoned = y_train.copy()
    class1_indices = np.where(y_train == 1)[0]
    n_class1 = len(class1_indices)

    n_flip = int(n_class1 * flip_fraction)
    if n_flip == 0:
        return y_poisoned, np.array([], dtype=int), np.array([], dtype=int)

    n_to_0 = int(n_flip * ratio_to_class0)
    n_to_2 = n_flip - n_to_0

    rng = np.random.default_rng(seed)
    rng.shuffle(class1_indices)

    idx_to_0 = class1_indices[:n_to_0]
    idx_to_2 = class1_indices[n_to_0 : n_to_0 + n_to_2]

    y_poisoned[idx_to_0] = 0
    y_poisoned[idx_to_2] = 2
    return y_poisoned, idx_to_0, idx_to_2


def train_ovr_model(X_train, y_train_poisoned, seed):
    # Keep the exact hyperparameters expected by the challenge notebook/API.
    base_estimator = LogisticRegression(
        random_state=seed,
        solver="liblinear",
        C=1.0,
        max_iter=200,
    )
    model = OneVsRestClassifier(base_estimator)
    model.fit(X_train, y_train_poisoned)
    return model


def local_proxy_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    row1 = cm[1]
    total = int(row1.sum())

    c10 = row1[0] / total if total else 0.0
    c11 = row1[1] / total if total else 0.0
    c12 = row1[2] / total if total else 0.0
    c13 = row1[3] / total if total else 0.0

    class3_recall = recall_score(y_test, y_pred, labels=[3], average="macro", zero_division=0)

    return {
        "class1_to_0": c10,
        "class1_to_1": c11,
        "class1_to_2": c12,
        "class1_to_3": c13,
        "class3_recall": float(class3_recall),
    }


def save_model_params_npz(model, out_path):
    params = {}
    for i, estimator in enumerate(model.estimators_):
        params[f"coef_estimator_{i}"] = estimator.coef_
        params[f"intercept_estimator_{i}"] = estimator.intercept_
    params["classes_"] = model.classes_
    np.savez_compressed(out_path, **params)


def submit_model(base_url, evaluate_path, model_file_path, timeout):
    url = base_url.rstrip("/") + evaluate_path
    with open(model_file_path, "rb") as f:
        files = {
            "model_params": (Path(model_file_path).name, f, "application/octet-stream"),
        }
        response = requests.post(url, files=files, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"raw_response": response.text, "status_code": response.status_code}


def ping_health(base_url, health_path, timeout):
    if not health_path:
        return None

    url = base_url.rstrip("/") + health_path
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"raw_response": response.text, "status_code": response.status_code}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Solve final skills assessment by poisoning labels for a 4-class OvR Logistic Regression model."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Evaluator base URL.")
    parser.add_argument(
        "--evaluate-path",
        default=DEFAULT_EVALUATE_PATH,
        help="Path for model evaluation endpoint.",
    )
    parser.add_argument(
        "--health-path",
        default=DEFAULT_HEALTH_PATH,
        help="Health endpoint path. Use '' to disable health check.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name(DEFAULT_DATASET)),
        help="Path to assessment_dataset.npz",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for API requests.",
    )
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Submit only one config from --flip-fraction and --ratio-to-0.",
    )
    parser.add_argument(
        "--flip-fraction",
        type=float,
        default=0.90,
        help="Fraction of class-1 labels to flip.",
    )
    parser.add_argument(
        "--ratio-to-0",
        type=float,
        default=0.30,
        help="Among flipped labels, fraction to send from class 1 to class 0 (rest go to class 2).",
    )
    parser.add_argument(
        "--keep-model-file",
        action="store_true",
        help="Keep generated .npz model files instead of using temp files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    print(f"Loaded dataset: X_train={X_train.shape}, y_train={y_train.shape}")

    if args.health_path:
        try:
            health = ping_health(args.base_url, args.health_path, args.timeout)
            print("Health response:")
            print(json.dumps(health, indent=2))
        except Exception as exc:
            print(f"Health check failed (continuing anyway): {exc}")

    configs = (
        [(args.flip_fraction, args.ratio_to_0)]
        if args.single_config
        else CANDIDATE_CONFIGS
    )

    print(f"Trying {len(configs)} candidate poisoning configurations...")

    for i, (flip_fraction, ratio_to_0) in enumerate(configs, start=1):
        y_poisoned, idx_to_0, idx_to_2 = poison_labels(
            y_train,
            flip_fraction=flip_fraction,
            ratio_to_class0=ratio_to_0,
            seed=args.seed,
        )

        model = train_ovr_model(X_train, y_poisoned, args.seed)
        metrics = local_proxy_metrics(model, X_test, y_test)

        print(
            f"\n[{i}/{len(configs)}] flip_fraction={flip_fraction:.3f}, ratio_to_0={ratio_to_0:.3f}"
        )
        print(f"  Flipped class 1 -> 0: {len(idx_to_0)}")
        print(f"  Flipped class 1 -> 2: {len(idx_to_2)}")
        print(
            "  Local proxy: "
            f"C1->0={metrics['class1_to_0']:.3f}, "
            f"C1->1={metrics['class1_to_1']:.3f}, "
            f"C1->2={metrics['class1_to_2']:.3f}, "
            f"C1->3={metrics['class1_to_3']:.3f}, "
            f"Class3Recall={metrics['class3_recall']:.3f}"
        )

        if args.keep_model_file:
            out_path = Path(__file__).with_name(
                f"poisoned_model_f{flip_fraction:.3f}_r{ratio_to_0:.3f}.npz"
            )
            save_model_params_npz(model, out_path)
            model_file = str(out_path)
        else:
            with NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                model_file = tmp.name
            save_model_params_npz(model, model_file)

        try:
            result = submit_model(
                args.base_url,
                args.evaluate_path,
                model_file,
                timeout=args.timeout,
            )
            print("  API response:")
            print(json.dumps(result, indent=2))

            if result.get("attack_successful") or result.get("success"):
                if result.get("flag"):
                    print(f"\nFLAG: {result['flag']}")
                else:
                    print("\nSuccess condition reported by API, but no flag field found.")
                return

        except Exception as exc:
            print(f"  Submission failed: {exc}")

        finally:
            if not args.keep_model_file:
                try:
                    Path(model_file).unlink(missing_ok=True)
                except Exception:
                    pass

    print("\nNo successful configuration found in this run.")
    print("Try tweaking --seed or run with --single-config and custom fractions.")


if __name__ == "__main__":
    main()
