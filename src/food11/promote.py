"""
Quality gate: promote a registered food11 model version to `champion` only if
it beats the current champion by more than two standard errors.

Usage:
    uv run python src/food11/promote.py --version 8
"""

from __future__ import annotations

import argparse
import math
import os

from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "food11"
METRIC = "test_accuracy"
# Size of the mini evaluation split (data/food11_processed_mini/evaluation).
N_TEST = 1096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the quality gate to a model version")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="size of the test split")
    return parser.parse_args()


def accuracy_of(client: MlflowClient, run_id: str) -> float:
    return client.get_run(run_id).data.metrics[METRIC]


def main() -> None:
    args = parse_args()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    candidate = client.get_model_version(MODEL_NAME, args.version)
    acc_candidate = accuracy_of(client, candidate.run_id)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        champion = None

    if champion is None:
        # Nothing to compare with: the first model becomes the champion.
        decision = "promoted"
        acc_champion = float("nan")
        threshold = float("nan")
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
    elif champion.version == candidate.version:
        print(f"Version {candidate.version} is already the champion, nothing to do.")
        return
    else:
        acc_champion = accuracy_of(client, champion.run_id)
        se = math.sqrt(acc_champion * (1 - acc_champion) / args.n_test)
        threshold = acc_champion + 2 * se
        print(f"champion v{champion.version}: {acc_champion:.4f} (se={se:.4f})")
        print(f"candidate v{candidate.version}: {acc_candidate:.4f}, needs >= {threshold:.4f}")

        if acc_candidate >= threshold:
            decision = "promoted"
            client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        else:
            decision = "rejected"
            client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)

    client.set_model_version_tag(MODEL_NAME, candidate.version, "gate_decision", decision)
    client.set_model_version_tag(MODEL_NAME, candidate.version, "candidate_accuracy", f"{acc_candidate:.4f}")
    client.set_model_version_tag(MODEL_NAME, candidate.version, "champion_accuracy", f"{acc_champion:.4f}")
    client.set_model_version_tag(MODEL_NAME, candidate.version, "gate_threshold", f"{threshold:.4f}")
    print(f"Decision for version {candidate.version}: {decision}")


if __name__ == "__main__":
    main()
