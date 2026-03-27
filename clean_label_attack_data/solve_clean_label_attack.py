import argparse
import json
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors


DATASET_FILENAME = "clean_label_eval_dataset.npz"
DEFAULT_EVALUATOR_URL = "http://154.57.164.75:30159"
DEFAULT_NEIGHBORS = 5
DEFAULT_EPSILON = 0.8
DEFAULT_SEED = 1337


def infer_attack_classes(target_class):
    if target_class == 2:
        return 1, 1
    if target_class == 0:
        return 1, 1
    if target_class == 1:
        return 0, 0
    raise ValueError(f"Unexpected target class: {target_class}")


def load_dataset(dataset_path):
    data = np.load(dataset_path)
    return (
        data["Xtr"],
        data["ytr"],
        data["Xte"],
        data["yte"],
        int(data["target_idx"].item()),
    )


def train_ovr_model(X_train, y_train, seed):
    model = OneVsRestClassifier(
        LogisticRegression(random_state=seed, C=1.0, solver="liblinear")
    )
    model.fit(X_train, y_train)
    return model


def perform_clean_label_attack(
    X_train_orig,
    y_train_orig,
    target_idx,
    target_class,
    perturb_class,
    n_neighbors,
    epsilon_cross,
    seed,
):
    baseline_model = train_ovr_model(X_train_orig, y_train_orig, seed)
    weights = np.vstack([est.coef_[0] for est in baseline_model.estimators_])
    intercepts = np.array([est.intercept_[0] for est in baseline_model.estimators_])

    boundary_vector = weights[perturb_class] - weights[target_class]
    boundary_norm = np.linalg.norm(boundary_vector)
    if boundary_norm < 1e-9:
        raise ValueError("Boundary vector norm is too small to determine a push direction.")

    unit_push_direction = -boundary_vector / boundary_norm
    perturbation_vector = epsilon_cross * unit_push_direction

    perturb_candidate_indices = np.where(y_train_orig == perturb_class)[0]
    if len(perturb_candidate_indices) == 0:
        raise ValueError(f"No samples found for perturbing class {perturb_class}.")

    n_neighbors = min(n_neighbors, len(perturb_candidate_indices))
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be at least 1.")

    neighbor_finder = NearestNeighbors(n_neighbors=n_neighbors)
    neighbor_finder.fit(X_train_orig[perturb_candidate_indices])
    _, relative_neighbor_indices = neighbor_finder.kneighbors(
        X_train_orig[target_idx].reshape(1, -1)
    )
    perturbed_indices = perturb_candidate_indices[relative_neighbor_indices.flatten()]

    X_train_poisoned = X_train_orig.copy()
    y_train_poisoned = y_train_orig.copy()
    X_train_poisoned[perturbed_indices] = (
        X_train_poisoned[perturbed_indices] + perturbation_vector
    )

    return (
        baseline_model,
        X_train_poisoned,
        y_train_poisoned,
        perturbed_indices,
        perturbation_vector,
        intercepts[perturb_class] - intercepts[target_class],
    )


def extract_submission_params(model):
    return {
        "weights": [est.coef_[0].tolist() for est in model.estimators_],
        "intercept": [est.intercept_[0] for est in model.estimators_],
    }


def submit_model(base_url, payload):
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    response = requests.post(f"{base_url}/evaluate", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the clean-label evaluator.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_EVALUATOR_URL,
        help="Evaluator base URL.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name(DATASET_FILENAME)),
        help="Path to the clean-label dataset .npz file.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
        help="Number of nearest perturbing-class neighbors to modify.",
    )
    parser.add_argument(
        "--epsilon-cross",
        type=float,
        default=DEFAULT_EPSILON,
        help="Magnitude of the cross-boundary perturbation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for model training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test, target_index = load_dataset(args.dataset)
    target_class = int(y_train[target_index])
    perturbing_class, misclassify_as_class = infer_attack_classes(target_class)

    (
        baseline_model,
        X_train_poisoned,
        y_train_poisoned,
        perturbed_indices,
        perturbation_vector,
        _,
    ) = perform_clean_label_attack(
        X_train,
        y_train,
        target_index,
        target_class,
        perturbing_class,
        args.n_neighbors,
        args.epsilon_cross,
        args.seed,
    )

    poisoned_model = train_ovr_model(X_train_poisoned, y_train_poisoned, args.seed)
    target_prediction = int(poisoned_model.predict(X_train[target_index].reshape(1, -1))[0])
    clean_accuracy = accuracy_score(y_test, poisoned_model.predict(X_test))

    print(f"Target index: {target_index}")
    print(f"Target class: {target_class}")
    print(f"Required misclassification class: {misclassify_as_class}")
    print(f"Perturbing class: {perturbing_class}")
    print(f"Perturbed indices: {perturbed_indices.tolist()}")
    print(f"Perturbation vector: {perturbation_vector.tolist()}")
    print(f"Target prediction after poisoning: {target_prediction}")
    print(f"Clean test accuracy: {clean_accuracy:.4f}")
    print(
        f"Baseline target prediction: {int(baseline_model.predict(X_train[target_index].reshape(1, -1))[0])}"
    )

    result = submit_model(args.base_url, extract_submission_params(poisoned_model))
    print(json.dumps(result, indent=2))
    if result.get("success") and result.get("flag"):
        print(f"Flag: {result['flag']}")


if __name__ == "__main__":
    main()