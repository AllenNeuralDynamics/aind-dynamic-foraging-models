"""Every module the subpackage imports must be a declared dependency.

Four separate CI and production failures in this work traced to the same cause: a module
present in a development environment but absent from the declared set, so the code ran
locally and failed on a clean install. This test closes that gap by construction.
"""

import ast
import pathlib
import re
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "aind_dynamic_foraging_models" / "hierarchical_bayes"

# Modules that ship with Python, and the package's own namespace.
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"aind_dynamic_foraging_models"}


def _imported_top_level_modules(directory):
    """Top-level module names imported by the .py files directly in ``directory``."""
    found = set()
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {name for name in found if name not in STDLIB}


def _declared_dependencies():
    """Distribution names from the core dependencies plus the bayes extra."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        raise unittest.SkipTest("needs tomllib")

    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    project = config["project"]
    names = set(project.get("dependencies", []))
    names |= set(project.get("optional-dependencies", {}).get("bayes", []))
    # A requirement string is name + optional extras + optional version specifier; take
    # the leading name. Splitting on a hand-listed set of operators misses one sooner or
    # later -- this test failed on `numpyro<0.20` for exactly that reason.
    return {re.match(r"[A-Za-z0-9._-]+", n.strip()).group(0) for n in names if n.strip()}


class TestDeclaredDependencies(unittest.TestCase):
    """The bayes extra must cover what the subpackage actually imports."""

    def test_every_import_is_declared(self):
        """No module is imported that a clean install would not provide."""
        imported = _imported_top_level_modules(PACKAGE)
        declared = _declared_dependencies()
        # scikit-learn imports as sklearn; nothing else here differs from its dist name.
        aliases = {"sklearn": "scikit-learn"}
        missing = {
            name for name in imported
            if aliases.get(name, name) not in declared
        }
        self.assertEqual(
            missing, set(),
            f"imported but not declared in pyproject: {sorted(missing)}. "
            "A development environment that happens to have these would hide the gap.",
        )

    def test_bayes_extra_is_not_empty(self):
        """The extra exists and carries the sampler stack."""
        declared = _declared_dependencies()
        for required in ("jax", "numpyro", "arviz"):
            self.assertIn(required, declared)


if __name__ == "__main__":
    unittest.main()
