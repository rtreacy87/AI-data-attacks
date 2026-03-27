import argparse
import json
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression


DATASET_FILENAME = "label_flipping_dataset.npz"
DEFAULT_EVALUATOR_URL = "http://154.57.164.78:31332"
DEFAULT_TARGET_CLASS = 0
DEFAULT_NEW_LABEL = 1
DEFAULT_POISON_FRACTION = 0.55
DEFAULT_SEED = 1337


def targeted_class_label_flip(y_train, target_class, new_label, poison_fraction, seed):
    if not 0 <= poison_fraction <= 1:
        raise ValueError("poison_fraction must be between 0 and 1.")
    if target_class == new_label:
        raise ValueError("target_class and new_label cannot be the same.")

    unique_labels = np.unique(y_train)
    if target_class not in unique_labels:
        raise ValueError(f"target_class ({target_class}) does not exist in y_train.")
    if new_label not in unique_labels:
        raise ValueError(f"new_label ({new_label}) does not exist in y_train.")

    target_indices = np.where(y_train == target_class)[0]
    n_target_samples = len(target_indices)
    if n_target_samples == 0:
        return y_train.copy(), np.array([], dtype=int)

    n_to_flip = int(n_target_samples * poison_fraction)
    if n_to_flip == 0:
        return y_train.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    indices_within_target_set = rng.choice(n_target_samples, size=n_to_flip, replace=False)
    flipped_indices = target_indices[indices_within_target_set]

    y_train_poisoned = y_train.copy()
    y_train_poisoned[flipped_indices] = new_label
    return y_train_poisoned, flipped_indices


def load_dataset(dataset_path):
    data = np.load(dataset_path)
    return data["Xtr"], data["ytr"], data["Xte"], data["yte"]


def train_model(X_train, y_train_poisoned, seed):
    model = LogisticRegression(random_state=seed, solver="liblinear")
    model.fit(X_train, y_train_poisoned)
    return model


def submit_model(base_url, model):
    payload = {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    response = requests.post(f"{base_url}/evaluate_targeted", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the targeted label attack evaluator.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_EVALUATOR_URL,
        help="Evaluator base URL.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name(DATASET_FILENAME)),
        help="Path to the label flipping dataset .npz file.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=DEFAULT_TARGET_CLASS,
        help="Class to poison.",
    )
    parser.add_argument(
        "--new-label",
        type=int,
        default=DEFAULT_NEW_LABEL,
        help="Label assigned to poisoned samples.",
    )
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=DEFAULT_POISON_FRACTION,
        help="Fraction of target-class samples to relabel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for sample selection and model training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, _, _ = load_dataset(args.dataset)
    y_train_poisoned, flipped_indices = targeted_class_label_flip(
        y_train,
        target_class=args.target_class,
        new_label=args.new_label,
        poison_fraction=args.poison_fraction,
        seed=args.seed,
    )

    print(f"Poisoned labels shape: {y_train_poisoned.shape}")
    print(f"Number of targeted labels flipped: {len(flipped_indices)}")

    model = train_model(X_train, y_train_poisoned, args.seed)
    result = submit_model(args.base_url, model)

    print(json.dumps(result, indent=2))
    if result.get("success") and result.get("flag"):
        print(f"Flag: {result['flag']}")


if __name__ == "__main__":
    main()