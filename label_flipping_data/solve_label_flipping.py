import argparse
import json
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression


DATASET_FILENAME = "label_flipping_dataset.npz"
DEFAULT_EVALUATOR_URL = "http://154.57.164.78:32487"
DEFAULT_POISON_RATE = 0.60
DEFAULT_SEED = 1337


def flip_labels(y, poison_percentage, seed):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")

    n_samples = len(y)
    n_to_flip = int(n_samples * poison_percentage)
    if n_to_flip == 0:
        return y.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    flipped_indices = rng.choice(n_samples, size=n_to_flip, replace=False)

    y_poisoned = y.copy()
    y_poisoned[flipped_indices] = np.where(y_poisoned[flipped_indices] == 0, 1, 0)
    return y_poisoned, flipped_indices


def load_dataset(dataset_path):
    data = np.load(dataset_path)
    return data["Xtr"], data["ytr"], data["Xte"], data["yte"]


def train_model(X_train, y_train_poisoned, seed):
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train_poisoned)
    return model


def submit_model(base_url, model):
    payload = {
        "weights": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    response = requests.post(f"{base_url}/evaluate", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the label flipping evaluator.")
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
        "--poison-rate",
        type=float,
        default=DEFAULT_POISON_RATE,
        help="Fraction of labels to flip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for label selection and model training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, _, _ = load_dataset(args.dataset)
    y_train_poisoned, flipped_indices = flip_labels(y_train, args.poison_rate, args.seed)

    print(f"Poisoned labels shape: {y_train_poisoned.shape}")
    print(f"Number of labels flipped: {len(flipped_indices)}")

    model = train_model(X_train, y_train_poisoned, args.seed)
    result = submit_model(args.base_url, model)

    print(json.dumps(result, indent=2))
    if result.get("success") and result.get("flag"):
        print(f"Flag: {result['flag']}")


if __name__ == "__main__":
    main()