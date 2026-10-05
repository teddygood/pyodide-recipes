import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text())
config = next(
    node.value.args[0]
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "CONFIG"
        for target in node.targets
    )
)
dependencies = next(
    value
    for key, value in zip(config.keys, config.values, strict=True)
    if isinstance(key, ast.Constant) and key.value == "Build Dependencies"
)
deps = eval(
    compile(ast.Expression(dependencies), "<numpy config>", "eval"),
    {"__builtins__": {}, "bool": bool},
)
backend = sys.argv[2] if len(sys.argv) > 2 else "blis"
expected_name = "openblas-experiment" if backend == "openblas" else backend
if backend not in ("none", "blis", "openblas"):
    raise SystemExit("Unknown BLAS backend")
if (
    backend != "none"
    and (deps["blas"]["name"] != expected_name or not deps["blas"]["found"])
) or (backend == "none" and deps["blas"]["found"]):
    raise SystemExit("Unexpected NumPy BLAS configuration")
if deps["lapack"]["found"]:
    raise SystemExit("Expected NumPy's internal LAPACK")
print(f"NumPy BLAS backend: {backend}; LAPACK uses the internal implementation")
