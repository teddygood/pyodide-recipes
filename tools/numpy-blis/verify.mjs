import { verifyRuntime } from "./verify_runtime.mjs";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const [outputDir = ".numpy-blis"] = process.argv.slice(2);
const out = resolve(outputDir);
const {variant, backend} = JSON.parse(readFileSync(resolve(out, "variant.json")));
const dist = resolve(out, "dist");
const { loadPyodide } = await import(pathToFileURL(resolve(dist, "pyodide.mjs")));
const py = await loadPyodide({ indexURL: dist });
await py.loadPackage("numpy");
const probeBytes = backend === "blis" ? readFileSync(resolve(out, "dispatch_probe.so")) : null;
const correctness = readFileSync(fileURLToPath(new URL("./correctness.py", import.meta.url)), "utf8");
const result = await verifyRuntime(py, {variant, backend}, probeBytes, correctness);
writeFileSync(resolve(out, "verification.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({variant, dispatch: result.dispatch, correctness: result.correctness}, null, 2));
if (result.correctness.failures.length) process.exitCode = 1;
