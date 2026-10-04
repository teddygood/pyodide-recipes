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
py.globals.set("expected_backend", backend);
const result = JSON.parse(py.runPython(`
import json, sys
import numpy as np
assert np.__version__ == "2.4.6"
config = np.show_config(mode="dicts")
blas = config["Build Dependencies"]["blas"]
if expected_backend == "blis":
    assert blas["name"] == "blis" and blas["found"]
else:
    assert not blas.get("found", False)
assert not config["Build Dependencies"]["lapack"].get("found", False)
json.dumps({"numpy": np.__version__, "python": sys.version, "config": config})
`));
result.variant = variant;
if (variant !== "noblas") {
  const module = py._module;
  const libdir = "/lib/python3.15/site-packages/numpy.libs";
  const libraries = module.FS.readdir(libdir).filter(name => name.startsWith("libblis"));
  if (libraries.length !== 1) throw new Error("Expected one vendored BLIS library");
  result.blis_library = libraries[0];
  await module.loadDynamicLibrary(`${libdir}/${libraries[0]}`,
    { global: true, nodelete: true, loadAsync: true });
  module.FS.writeFile("/tmp/numpy_blis_probe.so", readFileSync(resolve(out, "dispatch_probe.so")));
  const probe = {};
  await module.loadDynamicLibrary("/tmp/numpy_blis_probe.so",
    { global: false, nodelete: true, loadAsync: true }, probe);
  if (!probe.probe_install()) throw new Error("No selected DGEMM kernel");
  result.dispatch = {
    selected_is_simd: Boolean(probe.probe_selected_is_simd()),
    mr: probe.probe_mr(), nr: probe.probe_nr(), calls: {},
  };
  try {
    if (result.dispatch.selected_is_simd !== (variant === "simd") ||
        result.dispatch.mr !== 4 || result.dispatch.nr !== (variant === "simd" ? 4 : 8)) {
      throw new Error("Unexpected DGEMM kernel");
    }
    py.runPython("a=np.ones((256,256)); b=np.ones((256,256))");
    for (const [name, code] of Object.entries({matmul: "a @ b", dot: "np.dot(a,b)"})) {
      probe.probe_reset();
      py.runPython(code);
      const calls = probe.probe_calls();
      result.dispatch.calls[name] = calls;
      if (calls === 0) throw new Error(`${name} did not call the selected kernel`);
    }
  } finally {
    probe.probe_restore();
  }
}
const script = fileURLToPath(new URL("./correctness.py", import.meta.url));
py.runPython(readFileSync(script, "utf8"));
result.correctness = JSON.parse(py.globals.get("correctness_json"));
writeFileSync(resolve(out, "verification.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({variant, dispatch: result.dispatch, correctness: result.correctness}, null, 2));
if (result.correctness.failures.length) process.exitCode = 1;
