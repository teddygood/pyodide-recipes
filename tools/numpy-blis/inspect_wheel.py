import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from auditwheel_emscripten.emscripten_tools.webassembly import Module

out = Path(sys.argv[1])
(wheel,) = (out / "dist").glob("numpy-*.whl")
assert wheel.name.endswith("cp315-cp315-pyemscripten_2026_5_wasm32.whl")
with ZipFile(wheel) as archive, TemporaryDirectory() as temp:
    libraries = [
        name
        for name in archive.namelist()
        if name.startswith("numpy.libs/libblis") and name.endswith(".so")
    ]
    assert len(libraries) == 1, libraries
    library = libraries[0]
    blis = archive.read(library)
    assert blis.startswith(bytes([0, 97, 115, 109])) and len(blis) > 100_000
    path = Path(temp) / "blis.so"
    path.write_bytes(blis)
    with Module(path) as module:
        exports = {item.name for item in module.get_exports()}
    assert {"cblas_dgemm", "bli_dgemm_wasm32_simd128_4x4"} <= exports
    (core,) = (
        name
        for name in archive.namelist()
        if "/_multiarray_umath." in name and name.endswith(".so")
    )
    path.write_bytes(archive.read(core))
    with Module(path) as module:
        needed = module.parse_dylink_section().needed
    assert Path(library).name in needed, needed
    result = {
        "wheel": wheel.name,
        "wheel_bytes": wheel.stat().st_size,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "vendored_blis": library,
        "blis_bytes": len(blis),
        "blis_sha256": hashlib.sha256(blis).hexdigest(),
        "core_needed": needed,
        "candidate_kernel_exported": True,
        "runtime_sha256": {
            name: hashlib.sha256((out / "dist" / name).read_bytes()).hexdigest()
            for name in (
                "pyodide.mjs",
                "pyodide.asm.mjs",
                "pyodide.asm.wasm",
                "python_stdlib.zip",
            )
        },
    }
(out / "artifacts.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
