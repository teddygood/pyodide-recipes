import json
import time

import numpy as np

manifest = json.loads(globals()["bench_manifest"])
results = []
for case in manifest["cases"]:
    rng = np.random.default_rng(manifest["seed"])
    a = rng.random((case["m"], case["k"])) - 0.5
    b = rng.random((case["k"], case["n"])) - 0.5
    dtype = np.dtype(case.get("dtype", manifest.get("dtype", "float64")))
    if dtype.kind == "c":
        a = a + 1j * (rng.random(a.shape) - 0.5)
        b = b + 1j * (rng.random(b.shape) - 0.5)
    a, b = a.astype(dtype), b.astype(dtype)
    if case["layout"] == "conjugate":
        a, b = a.conj().T.copy(order="C").T, b.conj().T.copy(order="C").T
    if case["layout"] == "F":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif case["layout"] == "transpose":
        a, b = a.T.copy(order="C").T, b.T.copy(order="C").T
    elif case["layout"] == "strided":
        aa = np.empty((a.shape[0] * 2, a.shape[1] * 2), dtype=dtype)
        bb = np.empty((b.shape[0] * 2, b.shape[1] * 2), dtype=dtype)
        aa[::2, ::2], bb[::2, ::2] = a, b
        a, b = aa[::2, ::2], bb[::2, ::2]
    operation = np.matmul if case["operation"] == "matmul" else np.dot
    expected = np.einsum("ik,kj->ij", a, b, optimize=False)
    tolerance = 3e-4 if dtype.itemsize in (4, 8) and dtype != np.float64 else 1e-10
    np.testing.assert_allclose(
        operation(a, b), expected, rtol=tolerance, atol=tolerance
    )
    assert operation(a, b).dtype == dtype
    loops = 1
    while True:
        start = time.perf_counter()
        for _ in range(loops):
            operation(a, b)
        elapsed = time.perf_counter() - start
        if elapsed * 1000 >= manifest["batch_ms"]:
            break
        loops *= 2
    until = time.perf_counter() + manifest["warmup_ms"] / 1000
    while time.perf_counter() < until:
        for _ in range(loops):
            operation(a, b)
    samples = []
    for _ in range(manifest["samples"]):
        start = time.perf_counter()
        for _ in range(loops):
            operation(a, b)
        samples.append((time.perf_counter() - start) * 1000 / loops)
    results.append({"case": case["id"], "loops": loops, "samples_ms": samples})
benchmark_json = json.dumps(
    {
        "results": results,
        "numpy": np.__version__,
        "config": np.show_config(mode="dicts"),
    }
)
