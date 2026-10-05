import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const [outputDir, manifestFile, resultFile] = process.argv.slice(2);
const out = resolve(outputDir);
const variant = JSON.parse(readFileSync(resolve(out, "variant.json")));
const dist = resolve(out, "dist");
const { loadPyodide } = await import(pathToFileURL(resolve(dist, "pyodide.mjs")));
const py = await loadPyodide({ indexURL: dist });
await py.loadPackage("numpy");
py.globals.set("bench_manifest", readFileSync(manifestFile, "utf8"));
py.globals.set("expected_backend", variant.backend);
py.runPython(`import numpy as np
blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
assert np.__version__ == "2.4.6"
if expected_backend != "none":
    assert blas["name"] == ("openblas-experiment" if expected_backend == "openblas" else expected_backend) and blas["found"]
else:
    assert not blas.get("found", False)
`);
py.runPython(readFileSync(fileURLToPath(new URL("./benchmark.py", import.meta.url)), "utf8"));
const result = JSON.parse(py.globals.get("benchmark_json"));
result.variant = variant.variant;
result.node = process.version;
writeFileSync(resultFile, JSON.stringify(result, null, 2) + "\n");
console.log(`Saved ${resultFile}`);
