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
if deps["blas"]["name"] != "blis" or not deps["blas"]["found"]:
    raise SystemExit("NumPy did not find BLIS")
if deps["lapack"]["found"]:
    raise SystemExit("Expected NumPy's internal LAPACK")
print("NumPy found BLIS; LAPACK uses the internal implementation")
