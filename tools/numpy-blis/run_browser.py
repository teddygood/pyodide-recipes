import argparse
import functools
import http.server
import json
import os
import random
import statistics
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default=".numpy-blis")
parser.add_argument("--verify", action="store_true")
parser.add_argument(
    "--variants",
    nargs="+",
    default=["noblas", "reference", "simd", "openblas", "openblas-dev"],
)
args = parser.parse_args()
out = (root / args.output).resolve()
if not out.is_relative_to(root):
    raise SystemExit("Output must be inside the repository")
manifest = json.loads((root / "tools/numpy-blis/benchmark-manifest.json").read_text())
manifest["variants"] = args.variants
if args.verify:
    manifest["processes_per_variant"] = 1
manifest["schedule"] = [
    args.variants[r:] + args.variants[:r]
    for r in range(manifest["processes_per_variant"])
]
os.sched_setaffinity(0, {manifest["cpu_affinity"]})


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *_args):
        pass


server = http.server.ThreadingHTTPServer(
    ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"
results_dir = out / "browser"
results_dir.mkdir(exist_ok=True)
runs = []
try:
    with sync_playwright() as p:
        for round_id in range(1 if args.verify else manifest["processes_per_variant"]):
            current = dict(manifest)
            current["cases"] = manifest["cases"].copy()
            random.Random(manifest["seed"] + round_id).shuffle(current["cases"])
            variants = args.variants[round_id:] + args.variants[:round_id]
            for variant in variants:
                verification = json.loads(
                    (out / variant / "verification.json").read_text()
                )
                assert (
                    verification["correctness"]["passed"] == 1004
                    and not verification["correctness"]["failures"]
                )
                browser = p.chromium.launch(
                    headless=True, executable_path=os.environ.get("CHROMIUM_EXECUTABLE")
                )
                try:
                    page = browser.new_page()
                    page.goto(base + "/tools/numpy-blis/browser.html")
                    page.wait_for_function("typeof window.runExperiment === 'function'")
                    print(f"Chromium round {round_id + 1}: {variant}", flush=True)
                    result = page.evaluate(
                        "(options) => window.runExperiment(options)",
                        {
                            "output": base
                            + "/"
                            + str((out / variant).relative_to(root)),
                            "manifest": current,
                            "verify": args.verify,
                        },
                    )
                    result["browser"] = browser.version
                    result["user_agent"] = page.evaluate("navigator.userAgent")
                    result["manifest"] = current
                    name = (
                        f"verify-{variant}" if args.verify else f"{variant}-{round_id}"
                    )
                    (results_dir / f"{name}.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                    runs.append(result)
                finally:
                    browser.close()
finally:
    server.shutdown()
if not args.verify:
    artifacts = {
        variant: json.loads((out / variant / "artifacts.json").read_text())
        for variant in args.variants
    }
    assert (
        len(
            {
                json.dumps(a["runtime_sha256"], sort_keys=True)
                for a in artifacts.values()
            }
        )
        == 1
    )
    summary = {
        "manifest": manifest,
        "variants": args.variants,
        "artifacts": artifacts,
        "browser": runs[0]["browser"],
        "cases": [],
    }
    lines = [
        "| Case | " + " | ".join(v + " ms" for v in args.variants) + " |",
        "|---|" + "---:|" * len(args.variants),
    ]
    for case in manifest["cases"]:
        times = {}
        for variant in args.variants:
            medians = [
                statistics.median(item["samples_ms"])
                for run in runs
                if run["variant"] == variant
                for item in run["results"]
                if item["case"] == case["id"]
            ]
            times[variant] = {
                "process_medians_ms": medians,
                "median_ms": statistics.median(medians),
            }
        summary["cases"].append({"case": case["id"], "times": times})
        lines.append(
            "| "
            + case["id"]
            + " | "
            + " | ".join(f"{times[v]['median_ms']:.3f}" for v in args.variants)
            + " |"
        )
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
