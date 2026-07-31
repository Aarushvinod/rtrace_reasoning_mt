"""
make_random_pool_selections.py
──────────────────────────────
Generate the FIXED random-selection artifact for the random baseline:
for each devtest sentence, k_max unique uniformly-random indices into the
dataset's eng-side selection pool. Committed to the repo so every job LOADS
the same selection (no unseeded per-job regeneration, no create races).

Mirrors retrieval_helpers._load_or_create_random_pool_selections exactly
(same seed, same RNG consumption order), so a fallback regeneration on the
cluster reproduces this file byte-for-byte.

    python scripts/make_random_pool_selections.py wmt24pp   # 860-pool / 100-test
    python scripts/make_random_pool_selections.py flores    # 997-dev / 100-test

Output: data/random_pool_selections/<dataset>_eng_random_pool.json
"""

import json
import os
import sys

import numpy as np

SEED = 12345
K_MAX = 10  # max of the study's K_LIST (1,3,5,7,10)

DATASET_DIMS = {
    # dataset -> (n_dev = eng selection-pool size, n_devtest = fixed test size)
    "wmt24pp": (860, 100),
    "flores": (997, 100),
}


def main() -> None:
    dataset = (sys.argv[1] if len(sys.argv) > 1 else "wmt24pp").lower()
    if dataset not in DATASET_DIMS:
        raise SystemExit(f"dataset must be one of {sorted(DATASET_DIMS)}, got {dataset!r}")
    n_dev, n_devtest = DATASET_DIMS[dataset]

    rng = np.random.default_rng(SEED)
    selections = []
    for _ in range(n_devtest):
        chosen = rng.choice(n_dev, size=K_MAX, replace=False)
        selections.append([int(x) for x in chosen.tolist()])

    for i, row in enumerate(selections):
        assert len(set(row)) == K_MAX, f"row {i} has duplicates"
        assert all(0 <= x < n_dev for x in row), f"row {i} out of range"

    payload = {
        "n_dev": n_dev,
        "n_devtest": n_devtest,
        "k_max": K_MAX,
        "selections": selections,
    }

    out_path = os.path.join("data", "random_pool_selections", f"{dataset}_eng_random_pool.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"wrote {out_path}: {n_devtest} rows x k_max={K_MAX} from pool of {n_dev} (seed {SEED})")


if __name__ == "__main__":
    main()
