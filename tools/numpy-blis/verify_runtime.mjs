export async function verifyRuntime(py, {variant, backend}, probeBytes, correctness) {
py.globals.set("expected_backend", backend);
const result = JSON.parse(py.runPython(`
import json, sys
import numpy as np
assert np.__version__ == "2.4.6"
config = np.show_config(mode="dicts")
blas = config["Build Dependencies"]["blas"]
if expected_backend != "none":
    assert blas["name"] == ("openblas-experiment" if expected_backend == "openblas" else expected_backend) and blas["found"]
else:
    assert not blas.get("found", False)
assert not config["Build Dependencies"]["lapack"].get("found", False)
json.dumps({"numpy": np.__version__, "python": sys.version, "config": config})
`));
result.variant = variant;
if (backend === "blis") {
  const module = py._module;
  const libdir = "/lib/python3.15/site-packages/numpy.libs";
  const libraries = module.FS.readdir(libdir).filter(name => name.startsWith("libblis"));
  if (libraries.length !== 1) throw new Error("Expected one vendored BLIS library");
  result.blis_library = libraries[0];
  await module.loadDynamicLibrary(`${libdir}/${libraries[0]}`,
    { global: true, nodelete: true, loadAsync: true });
  module.FS.writeFile("/tmp/numpy_blis_probe.so", probeBytes);
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
if (backend === "openblas") {
  result.openblas = JSON.parse(py.runPython(`
import ctypes
from pathlib import Path
library, = Path("/lib/python3.15/site-packages/numpy.libs").glob("libopenblas*.so")
lib = ctypes.CDLL(str(library))
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_corename.restype = ctypes.c_char_p
config = lib.openblas_get_config().decode()
core = lib.openblas_get_corename().decode()
threads = lib.openblas_get_num_threads()
assert core == "WASM128_GENERIC" and threads == 1
json.dumps({"library": library.name, "config": config, "core": core, "threads": threads})
`));
}
py.runPython(correctness);
result.correctness = JSON.parse(py.globals.get("correctness_json"));
return result;
}
