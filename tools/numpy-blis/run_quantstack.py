import argparse
import contextlib
import functools
import http.server
import importlib.util
import json
import os
import random
import statistics
import subprocess
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", default=".numpy-blis")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
source = args.source.resolve()
pin = "749ef7ac2eb30e4675a718be527dd8c2b00ce1e2"
assert (
    subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    == pin
)
out = (root / args.output).resolve()
verification = json.loads((out / "simd/verification.json").read_text())
assert (
    verification["correctness"]["passed"] == 1004
    and not verification["correctness"]["failures"]
)
assert verification["dispatch"]["selected_is_simd"]
manifest = json.loads((root / "tools/numpy-blis/benchmark-manifest.json").read_text())
if args.smoke:
    manifest.update(
        cases=manifest["cases"][:1],
        samples=3,
        warmup_ms=100,
        batch_ms=5,
        processes_per_variant=1,
    )
manifest["variants"] = ["simd", "quantstack-034", "quantstack-dev"]
manifest["schedule"] = [
    manifest["variants"][r:] + manifest["variants"][:r]
    for r in range(manifest["processes_per_variant"])
]
spec = importlib.util.spec_from_file_location(
    "quantstack_runner", source / "run_host.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.WORK = out / "quantstack-work"
runner.WORK.mkdir(exist_ok=True)
results_dir = out / ("quantstack-smoke" if args.smoke else "quantstack")
results_dir.mkdir(exist_ok=True)
guest = (root / "tools/numpy-blis/benchmark.py").read_text()
packages = {}
packed = {}
for variant, env_name in (("quantstack-034", "ob034"), ("quantstack-dev", "obdev")):
    prefix = source / ".pixi/envs" / env_name
    records = [
        json.loads(p.read_text()) for p in (prefix / "conda-meta").glob("*.json")
    ]
    packages[variant] = [
        {k: r.get(k) for k in ("name", "version", "build", "sha256", "url")}
        for r in records
        if r["name"] in ("numpy", "openblas", "python", "pyjs")
    ]
    mount = results_dir / ("mount-" + env_name)
    mount.mkdir(exist_ok=True)
    script = """import json, sys
from pathlib import Path
import numpy as np
assert np.__version__ == "2.5.3"
blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
assert blas["found"] and "openblas" in blas["name"]
import ctypes
lib = ctypes.CDLL("/lib/libopenblas.so")
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_corename.restype = ctypes.c_char_p
openblas_config = lib.openblas_get_config().decode()
openblas_core = lib.openblas_get_corename().decode()
assert openblas_core == "WASM128_GENERIC" and lib.openblas_get_num_threads() == 1
bench_manifest = Path("/home/web_user/manifest.json").read_text()
"""
    script += (
        guest
        + '\nresult=json.loads(benchmark_json)\nresult["python"]=sys.version\nresult["runtime_openblas"]={"config":openblas_config,"core":openblas_core}\nprint("RESULT/"+json.dumps(result))\n'
    )
    (mount / "bench.py").write_text(script)
    (mount / "manifest.json").write_text(json.dumps(manifest))
    packed[variant] = runner.pack_and_serve(prefix, mount, "/home/web_user", "bench.py")

os.sched_setaffinity(0, {manifest["cpu_affinity"]})
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"
variants = ["simd", "quantstack-034", "quantstack-dev"]
runs = []
try:
    from pyjs_code_runner.backend.browser_main.server import server_context

    with contextlib.ExitStack() as stack:
        urls = {}
        for variant, (work, utils, port, html) in packed.items():
            _, url = stack.enter_context(server_context(work_dir=work, port=port))
            urls[variant] = (url + "/" + html, utils)
        with sync_playwright() as p:
            for round_id in range(manifest["processes_per_variant"]):
                current = dict(manifest)
                current["cases"] = manifest["cases"].copy()
                random.Random(manifest["seed"] + round_id).shuffle(current["cases"])
                for variant in variants[round_id:] + variants[:round_id]:
                    print(f"Chromium round {round_id + 1}: {variant}", flush=True)
                    browser = p.chromium.launch(
                        headless=True,
                        executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
                    )
                    try:
                        page = browser.new_page()
                        page.set_default_timeout(0)
                        if variant == "simd":
                            page.goto(base + "/tools/numpy-blis/browser.html")
                            page.wait_for_function(
                                "typeof window.runExperiment === 'function'"
                            )
                            result = page.evaluate(
                                "(options) => window.runExperiment(options)",
                                {
                                    "output": base
                                    + "/"
                                    + str((out / "simd").relative_to(root)),
                                    "manifest": current,
                                    "verify": False,
                                },
                            )
                        else:
                            url, utils = urls[variant]
                            received = []
                            errors = []

                            def on_console(message, received=received, errors=errors):
                                text = message.text
                                if "RESULT/" in text:
                                    received.append(
                                        json.loads(text.split("RESULT/", 1)[1])
                                    )
                                else:
                                    errors.append(text)

                            page.on("console", on_console)
                            page.goto(url)
                            status = page.evaluate(
                                """async (options) => {
"""
                                + utils
                                + """
const print = text => console.log(text);
const pyjs = await make_pyjs(print, print, true);
await pyjs.exec("from pathlib import Path; Path('/home/web_user/manifest.json').write_text(" + JSON.stringify(options.manifest) + ")");
return globalThis.eval_main_script(pyjs, "/home/web_user", "bench.py");
}""",
                                {"manifest": json.dumps(current)},
                            )
                            if int(status or 0) != 0 or len(received) != 1:
                                raise RuntimeError(
                                    f"Forge benchmark failed: {errors[-10:]}"
                                )
                            result = received[0]
                        result.update(
                            {
                                "variant": variant,
                                "browser": browser.version,
                                "manifest": current,
                            }
                        )
                        (results_dir / f"{variant}-{round_id}.json").write_text(
                            json.dumps(result, indent=2) + "\n"
                        )
                        runs.append(result)
                    finally:
                        browser.close()
finally:
    server.shutdown()
assert len({r["browser"] for r in runs}) == 1
summary = {
    "manifest": manifest,
    "browser": runs[0]["browser"],
    "source_commit": "749ef7ac2eb30e4675a718be527dd8c2b00ce1e2",
    "packages": packages,
    "cases": [],
}
for case in manifest["cases"]:
    times = {}
    for variant in variants:
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
(results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Saved " + str(results_dir / "summary.json"))
