import ast
from importlib.util import resolve_name
from pathlib import Path


def test_engine_and_evidence_import_boundaries():
    root = Path(__file__).resolve().parents[2]
    forbidden = ("spectra.api", "evaluation", "fastapi")
    violations = []
    for package in ("engine", "evidence"):
        for path in (root / "spectra" / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    parent = ".".join(path.relative_to(root).parts[:-1])
                    base = resolve_name("." * node.level + (node.module or ""), parent) if node.level else (
                        node.module or "")
                    modules = [base] + [base + "." + a.name for a in node.names]
                for module in modules:
                    banned = forbidden + (("spectra.evidence",) if package == "engine" else ())
                    if any(module == p or module.startswith(p + ".") for p in banned):
                        violations.append((str(path), module))
    assert not violations
