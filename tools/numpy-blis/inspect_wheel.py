import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from auditwheel_emscripten.emscripten_tools.webassembly import Module

out = Path(sys.argv[1])
variant_info = json.loads((out / "variant.json").read_text())
variant = variant_info["variant"]
backend = variant_info["backend"]
(wheel,) = (out / "dist").glob("numpy-*.whl")
assert wheel.name.endswith("cp315-cp315-pyemscripten_2026_5_wasm32.whl")
with ZipFile(wheel) as archive, TemporaryDirectory() as temp:
    libraries = [
        name
        for name in archive.namelist()
        if name.startswith(
            "numpy.libs/libblis" if backend == "blis" else "numpy.libs/libopenblas"
        )
        and name.endswith(".so")
    ]
    (core,) = (
        name
        for name in archive.namelist()
        if "/_multiarray_umath." in name and name.endswith(".so")
    )
    path = Path(temp) / "module.so"
    path.write_bytes(archive.read(core))
    with Module(path) as module:
        needed = module.parse_dylink_section().needed
    result = {
        "variant": variant,
        "wheel": wheel.name,
        "wheel_bytes": wheel.stat().st_size,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "core_needed": needed,
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
    if variant == "noblas":
        assert not libraries and not needed
    else:
        assert len(libraries) == 1, libraries
        library = libraries[0]
        blis = archive.read(library)
        assert blis.startswith(bytes([0, 97, 115, 109])) and len(blis) > 100_000
        path.write_bytes(blis)
        with Module(path) as module:
            exports = {item.name for item in module.get_exports()}
        assert "cblas_dgemm" in exports
        selected = "bli_dgemm_wasm32_simd128_4x4" in exports
        assert selected == (variant == "simd")
        if backend == "openblas":
            assert "openblas_get_config" in exports
        assert Path(library).name in needed, needed
        result.update(
            {
                "vendored_" + backend: library,
                backend + "_bytes": len(blis),
                backend + "_sha256": hashlib.sha256(blis).hexdigest(),
                "candidate_kernel_exported": selected,
            }
        )
(out / "artifacts.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
