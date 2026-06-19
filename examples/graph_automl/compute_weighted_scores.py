"""
Compute weighted scores from experiment_results_params_best.json and
experiment_results_params_default.json for all algorithm directories.

weighted_score = 0.8 * (test_acc or ari) + 0.2 * (1 / (1 + t * 2 / 300))

One CSV per dataset, rows = algorithms, columns = ws_run0, ws_run1, mean, std.
Output:
    graph_automl/weighted_scores/params_best/    <- from params_best.json
    graph_automl/weighted_scores/params_default/ <- from params_default.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

PERF_METRIC = {
    "cta": "test_acc",
    "domain": "ari",
}

RESULT_FILES = {
    "params_best": "experiment_results_params_best.json",
    "params_default": "experiment_results_params_default.json",
}


def get_perf_key(algo_name: str) -> str:
    for prefix, key in PERF_METRIC.items():
        if algo_name.startswith(prefix):
            return key
    raise ValueError(f"Unknown algorithm type: {algo_name}")


def weighted_score(perf: float, t: float) -> float:
    """Original: per-run score with 2× time penalty."""
    speed = 1 / (1 + t * 2 / 300)
    return 0.8 * perf + 0.2 * speed


def weighted_score_v2(perf_mean: float, t_total: float) -> float:
    """New: mean accuracy + summed time, no ×2 in penalty."""
    speed = 1 / (1 + t_total / 300)
    return 0.8 * perf_mean + 0.2 * speed


def process(result_filename: str, output_dir: Path, v2: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_records: dict[str, dict[str, tuple]] = {}

    for algo_dir in sorted(BASE_DIR.iterdir()):
        if not algo_dir.is_dir():
            continue
        results_file = algo_dir / result_filename
        if not results_file.exists():
            continue

        algo_name = algo_dir.name
        try:
            perf_key = get_perf_key(algo_name)
        except ValueError as e:
            print(f"  Skipping {algo_name}: {e}")
            continue

        with open(results_file) as f:
            data = json.load(f)

        for dataset, entries in data.items():
            entry = entries[0]
            runs = entry["runs"]

            if len(runs) < 2:
                print(f"  Warning: {algo_name}/{dataset} has fewer than 2 runs, skipping.")
                continue

            run0, run1 = runs[0], runs[1]

            if v2:
                perf_mean = (run0[perf_key] + run1[perf_key]) / 2
                t_total   = run0["time_seconds"] + run1["time_seconds"]
                ws = weighted_score_v2(perf_mean, t_total)
                dataset_records.setdefault(dataset, {})[algo_name] = (ws,)
            else:
                ws0 = weighted_score(run0[perf_key], run0["time_seconds"])
                ws1 = weighted_score(run1[perf_key], run1["time_seconds"])
                dataset_records.setdefault(dataset, {})[algo_name] = (ws0, ws1)

    for dataset, algo_scores in sorted(dataset_records.items()):
        rows = []
        for algo, vals in sorted(algo_scores.items()):
            if v2:
                rows.append({
                    "algorithm": algo,
                    "ws": vals[0],
                })
            else:
                ws0, ws1 = vals
                rows.append({
                    "algorithm": algo,
                    "ws_run0": ws0,
                    "ws_run1": ws1,
                    "mean": np.mean([ws0, ws1]),
                    "std": np.std([ws0, ws1]),
                })

        df = pd.DataFrame(rows).set_index("algorithm")
        out_path = output_dir / f"{dataset}.csv"
        df.to_csv(out_path)
        print(f"  Wrote {out_path.name}  ({len(df)} algorithms)")


for mode, filename in RESULT_FILES.items():
    output_dir = BASE_DIR / "weighted_scores" / mode
    print(f"\n=== {mode} -> {output_dir} ===")
    process(filename, output_dir)

for mode, filename in RESULT_FILES.items():
    output_dir = BASE_DIR / "weighted_scores" / f"{mode}_v2"
    print(f"\n=== {mode}_v2 -> {output_dir} ===")
    process(filename, output_dir, v2=True)

print("\nDone.")
