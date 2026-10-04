import json

import numpy as np

rng = np.random.default_rng(20261004)
passed = []
failures = []


def check(name, run, *args):
    try:
        run(*args)
        passed.append(name)
    except Exception as exc:
        failures.append({"case": name, "error": repr(exc)})


def layout(a, kind):
    if kind in ("C", "F"):
        return a.copy(order=kind)
    if kind == "transpose":
        return a.T.copy(order="C").T
    if kind == "strided":
        out = np.empty((a.shape[0] * 2, a.shape[1] * 2), dtype=a.dtype)
        out[::2, ::2] = a
        return out[::2, ::2]
    return a[::-1, ::-1]


def product(a, b):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            out[i, j] = sum(a[i, k].item() * b[k, j].item() for k in range(a.shape[1]))
    return out


def assert_product(operation, left, right, expected, tolerance):
    result = operation(left, right)
    assert result.shape == expected.shape and result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


def assert_strided_out(left, right, expected, tolerance):
    m, n = expected.shape
    storage = np.full((m * 2, n * 2), np.nan, dtype=np.float64)
    out = storage[::2, ::2]
    result = np.matmul(left, right, out=out)
    assert result is out
    np.testing.assert_allclose(out, expected, rtol=tolerance, atol=tolerance)
    assert np.isnan(storage[1::2, :]).all()
    assert np.isnan(storage[:, 1::2]).all()


shapes = [
    (0, 3, 5),
    (5, 0, 3),
    (5, 3, 0),
    (1, 1, 1),
    (1, 7, 3),
    (3, 7, 1),
    (3, 5, 7),
    (4, 4, 4),
    (5, 9, 7),
    (7, 5, 9),
    (16, 16, 16),
    (17, 17, 17),
    (33, 31, 17),
]
layouts = [
    ("C", "C"),
    ("C", "F"),
    ("F", "C"),
    ("F", "F"),
    ("transpose", "transpose"),
    ("strided", "strided"),
    ("negative", "negative"),
]
for dtype in (np.float64, np.float32, np.complex128, np.complex64, np.int64):
    tolerance = 3e-5 if dtype in (np.float32, np.complex64) else 5e-13
    for m, n, k in shapes:
        a = rng.integers(-8, 9, size=(m, k)).astype(dtype)
        b = rng.integers(-8, 9, size=(k, n)).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            a += 1j * rng.integers(-8, 9, size=a.shape)
            b += 1j * rng.integers(-8, 9, size=b.shape)
        elif dtype in (np.float32, np.float64):
            a /= 7
            b /= 7
        for order_a, order_b in layouts:
            left, right = layout(a, order_a), layout(b, order_b)
            expected = product(left, right)
            prefix = f"{np.dtype(dtype).name}/{m}x{n}x{k}/{order_a}-{order_b}"
            for name, operation in (("matmul", np.matmul), ("dot", np.dot)):
                check(
                    f"{prefix}/{name}",
                    assert_product,
                    operation,
                    left,
                    right,
                    expected,
                    tolerance,
                )
            if dtype == np.float64:
                check(
                    f"{prefix}/strided-out",
                    assert_strided_out,
                    left,
                    right,
                    expected,
                    tolerance,
                )


def check_batched():
    a = rng.random((2, 5, 7))
    b = rng.random((1, 7, 3))
    expected = np.stack([product(row, b[0]) for row in a])
    np.testing.assert_allclose(a @ b, expected, rtol=5e-13, atol=5e-13)


def check_alias():
    a = rng.random((9, 9))
    b = rng.random((9, 9))
    expected = product(a, b)
    np.matmul(a, b, out=a)
    np.testing.assert_allclose(a, expected, rtol=5e-13, atol=5e-13)


def check_linalg():
    a = rng.random((17, 17))
    spd = a.T @ a + 17 * np.eye(17)
    b = rng.random((17, 3))
    np.testing.assert_allclose(spd @ np.linalg.solve(spd, b), b, rtol=5e-13, atol=5e-13)
    chol = np.linalg.cholesky(spd)
    np.testing.assert_allclose(chol @ chol.T, spd, rtol=5e-13, atol=5e-13)
    q, r = np.linalg.qr(a)
    np.testing.assert_allclose(q @ r, a, rtol=5e-13, atol=5e-13)
    u, s, vt = np.linalg.svd(a)
    np.testing.assert_allclose((u * s) @ vt, a, rtol=5e-13, atol=5e-13)
    values, vectors = np.linalg.eigh(spd)
    np.testing.assert_allclose(spd @ vectors, vectors * values, rtol=5e-13, atol=5e-13)


check("batched-broadcast", check_batched)
check("alias-out", check_alias)
check("internal-lapack", check_linalg)
correctness_json = json.dumps({"passed": len(passed), "failures": failures})
