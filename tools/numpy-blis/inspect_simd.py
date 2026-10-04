import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

out = Path(sys.argv[1])
disassembler = Path(sys.argv[2])
(wheel,) = (out / "dist").glob("numpy-*.whl")
with ZipFile(wheel) as archive, TemporaryDirectory() as temp:
    (library,) = (
        name
        for name in archive.namelist()
        if name.startswith("numpy.libs/libblis") and name.endswith(".so")
    )
    data = archive.read(library)
    binary = Path(temp) / "blis.so"
    text = Path(temp) / "blis.wat"
    binary.write_bytes(data)
    subprocess.run([str(disassembler), str(binary), "-o", str(text)], check=True)
    code = text.read_text()
    kernels = {}
    for name, vector in {
        "bli_sgemm_wasm32_simd128_4x4": "f32x4",
        "bli_dgemm_wasm32_simd128_4x4": "f64x2",
        "bli_cgemm_wasm32_simd128_4x2": "f32x4",
        "bli_zgemm_wasm32_simd128_2x2": "f64x2",
    }.items():
        (label,) = re.findall(
            r'\(export "' + re.escape(name) + r'" \(func ([^ )]+)\)\)', code
        )
        start = re.search(r"\n \(func " + re.escape(label) + r"[\s)]", code).start()
        end = code.find("\n (func ", start + 1)
        body = code[start : end if end >= 0 else len(code) - 1]
        counts = {
            operation: body.count(operation)
            for operation in [
                vector + ".mul",
                vector + ".add",
                vector + ".sub",
                "i8x16.shuffle",
            ]
        }
        assert counts[vector + ".mul"] > 0 and counts[vector + ".add"] > 0
        assert "relaxed" not in body
        kernels[name] = {
            "instructions": counts,
            "function_sha256": hashlib.sha256(body.encode()).hexdigest(),
        }
result = {"library_sha256": hashlib.sha256(data).hexdigest(), "kernels": kernels}
(out / "kernel-instructions.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
