"""
Compute weighted scores from plain openevolve_output results.

For each run i in a results_*.json file:
    weighted_score_i = scores[i] * 0.8 + 0.2 * (1 / (1 + 2 * times[i] / 300))

Outputs one CSV per dataset to ./weighted_scores_openevolve/.
Each CSV has rows = algorithms, columns = weighted_score_0, weighted_score_1, mean, std.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EVOLO_DIR = Path(__file__).parent
OUTPUT_DIR = EVOLO_DIR / "weighted_scores_openevolve"
OUTPUT_DIR.mkdir(exist_ok=True)


def weighted_score(score: float, time: float, T: float = 300.0) -> float:
    return score * 0.8 + 0.2 * (1.0 / (1.0 + 2.0 * time / T))


def extract_algorithm(algo_dir: Path) -> str:
    """'cta_scdeepsort' or 'domain_spagcn' -> 'scdeepsort' / 'spagcn'"""
    name = algo_dir.name  # e.g. cta_scdeepsort
    for prefix in ("cta_", "domain_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_dataset(filename: str, algorithm: str) -> str:
    """
    Strip 'results_' prefix and '.json' suffix.
    If the remainder starts with '<algorithm>_', strip that prefix too.
    e.g. results_human_CD8_...json  (algorithm=scdeepsort) -> human_CD8_...
         results_efnst_151507.json  (algorithm=efnst)      -> 151507
    """
    stem = filename.removeprefix("results_").removesuffix(".json")
    prefix = algorithm + "_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem


# dataset_key -> {algorithm -> (ws0, ws1)}
data: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

for openevolve_dir in sorted(EVOLO_DIR.rglob("openevolve_output")):
    # Must be a direct child of an algo dir, not inside benchmarks or old
    if "old" in openevolve_dir.parts:
        continue
    if "benchmarks" in openevolve_dir.name:
        continue

    best_dir = openevolve_dir / "best"
    if not best_dir.is_dir():
        continue

    # Parent of openevolve_output is the algo dir (e.g. cta_scdeepsort)
    algo_dir = openevolve_dir.parent
    algorithm = extract_algorithm(algo_dir)

    for json_file in sorted(best_dir.glob("results_*.json")):
        dataset = extract_dataset(json_file.name, algorithm)

        with open(json_file) as f:
            result = json.load(f)

        scores = result["scores"]
        times = result["times"]

        if len(scores) < 2 or len(times) < 2:
            print(f"WARNING: {json_file} has fewer than 2 runs, skipping")
            continue

        ws0 = weighted_score(scores[0], times[0])
        ws1 = weighted_score(scores[1], times[1])

        data[dataset][algorithm] = (ws0, ws1, scores[0], scores[1], times[0], times[1])


def weighted_score_v2(perf_mean: float, t_total: float, T: float = 300.0) -> float:
    """New: mean accuracy + summed time, no ×2 penalty."""
    return perf_mean * 0.8 + 0.2 * (1.0 / (1.0 + t_total / T))


# Write one CSV per dataset (original formula)
for dataset, algo_scores in sorted(data.items()):
    rows = []
    for algorithm, (ws0, ws1, *_) in sorted(algo_scores.items()):
        mean = np.mean([ws0, ws1])
        std = np.std([ws0, ws1])
        rows.append({
            "algorithm": algorithm,
            "weighted_score_0": ws0,
            "weighted_score_1": ws1,
            "mean": mean,
            "std": std,
        })

    df = pd.DataFrame(rows).set_index("algorithm")
    out_path = OUTPUT_DIR / f"{dataset}.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({len(rows)} algorithms)")

print(f"\nDone. {len(data)} CSVs written to {OUTPUT_DIR}")

# Write one CSV per dataset (v2 formula: mean acc + summed time, no ×2)
OUTPUT_DIR_V2 = EVOLO_DIR / "weighted_scores_openevolve_v2"
OUTPUT_DIR_V2.mkdir(exist_ok=True)

for dataset, algo_scores in sorted(data.items()):
    rows = []
    for algorithm, (ws0, ws1, s0, s1, t0, t1) in sorted(algo_scores.items()):
        ws = weighted_score_v2((s0 + s1) / 2, t0 + t1)
        rows.append({"algorithm": algorithm, "ws": ws})

    df = pd.DataFrame(rows).set_index("algorithm")
    out_path = OUTPUT_DIR_V2 / f"{dataset}.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({len(rows)} algorithms)")

print(f"\nDone v2. {len(data)} CSVs written to {OUTPUT_DIR_V2}")
