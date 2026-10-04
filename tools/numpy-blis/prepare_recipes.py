import json
import shutil
import sys
from pathlib import Path

from ruamel.yaml import YAML

out = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
if variant not in (
    "noblas",
    "reference",
    "simd",
    "simd-all",
    "openblas",
    "openblas-dev",
):
    raise SystemExit(
        "Choose noblas, reference, simd, simd-all, openblas, or openblas-dev"
    )
if (out / "variant.json").exists():
    previous = json.loads((out / "variant.json").read_text())["variant"]
    if previous != variant:
        raise SystemExit("Use a separate output directory for each variant")
root = Path(__file__).parent
yaml = YAML()
backend = (
    "none"
    if variant == "noblas"
    else ("openblas" if variant.startswith("openblas") else "blis")
)
numpy = yaml.load((root / "recipes/numpy/meta.yaml").read_text())
numpy["build"]["post"] = numpy["build"]["post"].replace(
    "numpy/__config__.py", f"numpy/__config__.py {backend}"
)
if variant == "noblas":
    numpy.pop("requirements")
    numpy["build"][
        "script"
    ] = 'export PKG_CONFIG_LIBDIR="${WASM_LIBRARY_DIR}/share/pkgconfig"\nunset PKG_CONFIG_PATH\n'
    numpy["build"]["backend-flags"] = numpy["build"]["backend-flags"].replace(
        "-Dblas=blis", "-Dblas=none"
    )
if backend == "openblas":
    numpy["requirements"]["host"] = ["libopenblas"]
    numpy["build"]["script"] = (
        numpy["build"]["script"]
        .replace("share/pkgconfig", "lib/pkgconfig")
        .replace("blis", "openblas-experiment")
    )
    numpy["build"]["backend-flags"] = numpy["build"]["backend-flags"].replace(
        "-Dblas=blis", "-Dblas=openblas-experiment"
    )
recipes = out / "recipes"
(recipes / "numpy").mkdir(parents=True, exist_ok=True)
with (recipes / "numpy/meta.yaml").open("w") as file:
    yaml.dump(numpy, file)
if backend == "blis":
    blis = yaml.load((root / "recipes/libblis/meta.yaml").read_text())
    if variant == "simd":
        blis["source"] = {
            "url": "https://github.com/teddygood/blis/archive/9c32fa424cc48d988cfcd0a98176f1ac3c9ae817.tar.gz",
            "sha256": "a8a5d979432c2cecc60e143a0872f88de24972299fdb2208be9f52af95d1bcb2",
        }
    if variant == "reference":
        blis["source"] = {
            "url": "https://github.com/teddygood/blis/archive/12da45904d7bfeb926d9be1a62485a58c22cbc2c.tar.gz",
            "sha256": "7c53294526d1ca8de87f88bcda1535951f3924a4ebb0bc0219dbe844649d9656",
        }
    (recipes / "libblis").mkdir(exist_ok=True)
    with (recipes / "libblis/meta.yaml").open("w") as file:
        yaml.dump(blis, file)
if backend == "openblas":
    openblas = yaml.load((root / "recipes/libopenblas/meta.yaml").read_text())
    if variant == "openblas-dev":
        openblas["package"]["version"] = "0.3.35.dev0"
        openblas["source"] = {
            "url": "https://github.com/OpenMathLib/OpenBLAS/archive/539bb47f020d3278a05805f86c1ac63326b8a3d1.tar.gz",
            "sha256": "b3fffa95edf413afd7b5f032a88aeb46581515b6f2607295fcdb0ba072b4b637",
        }
    (recipes / "libopenblas").mkdir(exist_ok=True)
    with (recipes / "libopenblas/meta.yaml").open("w") as file:
        yaml.dump(openblas, file)
shutil.copyfile(root / "check_config.py", out / "check_config.py")
(out / "variant.json").write_text(
    json.dumps({"variant": variant, "backend": backend}) + "\n"
)
