import json
import shutil
import sys
from pathlib import Path

from ruamel.yaml import YAML

out = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
if variant not in ("noblas", "reference", "simd"):
    raise SystemExit("Choose noblas, reference, or simd")
if (out / "variant.json").exists():
    previous = json.loads((out / "variant.json").read_text())["variant"]
    if previous != variant:
        raise SystemExit("Use a separate output directory for each variant")
root = Path(__file__).parent
yaml = YAML()
backend = "none" if variant == "noblas" else "blis"
numpy = yaml.load((root / "recipes/numpy/meta.yaml").read_text())
numpy["build"]["post"] = numpy["build"]["post"].replace(
    "numpy/__config__.py", f"numpy/__config__.py {backend}"
)
if variant == "noblas":
    numpy.pop("requirements")
    numpy["build"]["script"] = (
        'export PKG_CONFIG_LIBDIR="${WASM_LIBRARY_DIR}/share/pkgconfig"\nunset PKG_CONFIG_PATH\n'
    )
    numpy["build"]["backend-flags"] = numpy["build"]["backend-flags"].replace(
        "-Dblas=blis", "-Dblas=none"
    )
recipes = out / "recipes"
(recipes / "numpy").mkdir(parents=True, exist_ok=True)
with (recipes / "numpy/meta.yaml").open("w") as file:
    yaml.dump(numpy, file)
if variant != "noblas":
    blis = yaml.load((root / "recipes/libblis/meta.yaml").read_text())
    if variant == "reference":
        blis["source"] = {
            "url": "https://github.com/teddygood/blis/archive/12da45904d7bfeb926d9be1a62485a58c22cbc2c.tar.gz",
            "sha256": "7c53294526d1ca8de87f88bcda1535951f3924a4ebb0bc0219dbe844649d9656",
        }
    (recipes / "libblis").mkdir(exist_ok=True)
    with (recipes / "libblis/meta.yaml").open("w") as file:
        yaml.dump(blis, file)
shutil.copyfile(root / "check_config.py", out / "check_config.py")
(out / "variant.json").write_text(
    json.dumps({"variant": variant, "backend": backend}) + "\n"
)
