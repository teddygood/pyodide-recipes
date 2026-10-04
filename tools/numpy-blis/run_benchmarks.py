import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".numpy-blis").resolve()
manifest = json.loads((root / "benchmark-manifest.json").read_text())
artifacts = {
    variant: json.loads((out / variant / "artifacts.json").read_text())
    for variant in manifest["variants"]
}
for variant in manifest["variants"]:
    verification = json.loads((out / variant / "verification.json").read_text())
    assert verification["correctness"]["passed"] == 1004
    assert not verification["correctness"]["failures"]
assert (
    len(
        {
            json.dumps(value["runtime_sha256"], sort_keys=True)
            for value in artifacts.values()
        }
    )
    == 1
)
if manifest["cpu_affinity"] not in os.sched_getaffinity(0):
    raise SystemExit("Requested CPU is unavailable")
results_dir = out / "benchmarks"
results_dir.mkdir(exist_ok=True)
runs = []
for round_id, variants in enumerate(manifest["schedule"]):
    current = dict(manifest)
    current["cases"] = manifest["cases"].copy()
    random.Random(manifest["seed"] + round_id).shuffle(current["cases"])
    manifest_path = results_dir / f"manifest-{round_id}.json"
    manifest_path.write_text(json.dumps(current, indent=2) + "\n")
    for variant in variants:
        result_path = results_dir / f"{variant}-{round_id}.json"
        print(f"Round {round_id + 1}: {variant}", flush=True)
        subprocess.run(
            [
                "taskset",
                "-c",
                str(manifest["cpu_affinity"]),
                "node",
                str(root / "benchmark.mjs"),
                str(out / variant),
                str(manifest_path),
                str(result_path),
            ],
            check=True,
        )
        result = json.loads(result_path.read_text())
        for case in result["results"]:
            case["median_ms"] = statistics.median(case["samples_ms"])
        result["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        runs.append(result)
summary = {"manifest": manifest, "artifacts": artifacts, "cases": []}
for case in manifest["cases"]:
    times = {}
    for variant in manifest["variants"]:
        medians = [
            item["median_ms"]
            for run in runs
            if run["variant"] == variant
            for item in run["results"]
            if item["case"] == case["id"]
        ]
        times[variant] = {
            "process_medians_ms": medians,
            "median_ms": statistics.median(medians),
        }
    reference = times["reference"]["median_ms"]
    candidate = times["simd"]["median_ms"]
    summary["cases"].append(
        {
            "case": case["id"],
            "times": times,
            "reference_over_simd": reference / candidate,
            "noblas_over_simd": times["noblas"]["median_ms"] / candidate,
        }
    )
(results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "| Case | No BLAS ms | Reference ms | SIMD ms | Reference/SIMD |",
    "|---|---:|---:|---:|---:|",
]
for case in summary["cases"]:
    t = case["times"]
    lines.append(
        f"| {case['case']} | {t['noblas']['median_ms']:.3f} | {t['reference']['median_ms']:.3f} | {t['simd']['median_ms']:.3f} | {case['reference_over_simd']:.2f}x |"
    )
(results_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
