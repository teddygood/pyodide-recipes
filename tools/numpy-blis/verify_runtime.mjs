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
  const {probe, library} = await loadBlisProbe(py, probeBytes);
  result.blis_library = library;
  if (!probe.probe_install()) throw new Error("No selected DGEMM kernel");
  result.dispatch = {
    selected_is_simd: Boolean(probe.probe_selected_is_simd()),
    mr: probe.probe_mr(), nr: probe.probe_nr(), calls: {},
  };
  try {
    if (result.dispatch.selected_is_simd !== (["simd", "simd-all"].includes(variant)) ||
        result.dispatch.mr !== 4 || result.dispatch.nr !== (["simd", "simd-all"].includes(variant) ? 4 : 8)) {
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
  result.floating_dispatch = {};
  const defaultMethods = [probe.probe_method_dt(2), probe.probe_method_dt(3)];
  result.default_complex_methods = defaultMethods.map(method => method === 0 ? "1m" : "native");
  const expectedMethods = variant === "simd-all" ? [0,0] : variant === "simd" ? [1,0] : [1,1];
  if (defaultMethods.some((method, dt) => method !== expectedMethods[dt]))
    throw new Error("Unexpected initialized complex method");
  for (const mode of ["default", "native", "1m"]) {
    probe.probe_set_methods(...defaultMethods);
    if (mode === "1m") probe.probe_set_native(0);
    if (!probe.probe_install_all(mode === "native" ? 1 : 0)) throw new Error("Missing GEMM kernel");
    const selected = [0, 1, 2, 3].map(dt => ({
      simd: Boolean(probe.probe_selected_dt(dt)),
      mr: probe.probe_mr_dt(dt), nr: probe.probe_nr_dt(dt),
    }));
    const expected = variant === "simd-all" ? [true, true, true, true]
      : [false, variant === "simd", false, false];
    if (selected.some((v, dt) => v.simd !== expected[dt])) throw new Error("Unexpected GEMM selection");
    if (variant === "simd-all" && selected.some((v, dt) =>
        v.mr !== [4,4,4,2][dt] || v.nr !== [4,4,2,2][dt])) throw new Error("Unexpected tile sizes");
    const methods = [probe.probe_method_dt(2), probe.probe_method_dt(3)];
    const dispatch = {selected, complex_methods: methods.map(method => method === 0 ? "1m" : "native"), operations: {}};
    try {
      for (const [dt, dtype] of ["float32", "float64", "complex64", "complex128"].entries()) {
        py.runPython(`a=np.full((128,128), 1+0.5j if "${dtype}".startswith("complex") else 1, dtype="${dtype}"); b=a.copy()`);
        for (const [name, code] of Object.entries({matmul: "a @ b", dot: "np.dot(a,b)"})) {
          probe.probe_reset_all();
          py.runPython(code);
          const counts = [0,1,2,3].map(index => probe.probe_calls_dt(index));
          dispatch.operations[dtype + "/" + name] = counts;
          const target = dt < 2 || methods[dt - 2] !== 0 ? dt : dt - 2;
          if (counts[target] === 0 || counts.some((n, i) => i !== target && n !== 0))
            throw new Error("Unexpected NumPy GEMM dispatch: " + dtype + "/" + mode + " " + JSON.stringify(counts) + " method=" + probe.probe_method_dt(dt));
        }
      }
    } finally {
      probe.probe_restore_all();
    }
    if (mode === "native") probe.probe_set_native(1);
    else if (mode === "1m") probe.probe_set_native(0);
    else probe.probe_set_methods(...defaultMethods);
    try {
      py.runPython(correctness);
      dispatch.correctness = JSON.parse(py.globals.get("correctness_json"));
      if (dispatch.correctness.failures.length) throw new Error("Floating correctness failed");
    } finally {
      probe.probe_set_methods(...defaultMethods);
    }
    result.floating_dispatch[mode] = dispatch;
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

export async function loadBlisProbe(py, probeBytes) {
  const module = py._module;
  const libdir = "/lib/python3.15/site-packages/numpy.libs";
  const libraries = module.FS.readdir(libdir).filter(name => name.startsWith("libblis"));
  if (libraries.length !== 1) throw new Error("Expected one vendored BLIS library");
  await module.loadDynamicLibrary(`${libdir}/${libraries[0]}`,
    { global: true, nodelete: true, loadAsync: true });
  module.FS.writeFile("/tmp/numpy_blis_probe.so", probeBytes);
  const probe = {};
  await module.loadDynamicLibrary("/tmp/numpy_blis_probe.so",
    { global: false, nodelete: true, loadAsync: true }, probe);
  return {probe, library: libraries[0]};
}
