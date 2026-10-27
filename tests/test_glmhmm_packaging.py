"""Every module the glm_hmm subpackage imports must be a declared dependency.

Same contract as ``test_hb_packaging``: a module present in a development environment but
missing from the declared set runs locally and fails on a clean install.
"""

import ast
import pathlib
import re
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "aind_dynamic_foraging_models" / "glm_hmm"
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"aind_dynamic_foraging_models"}
# Import names that differ from their distribution names.
ALIASES = {"tensorflow_probability": "tfp-nightly", "sklearn": "scikit-learn"}


def _imported_top_level_modules(directory):
    """Top-level module names imported by the .py files directly in ``directory``."""
    found = set()
    for path in sorted(directory.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {name for name in found if name not in STDLIB}


def _declared(extra="glmhmm"):
    """Distribution names from the core dependencies plus ``extra``."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        raise unittest.SkipTest("needs tomllib")
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    names = set(project.get("dependencies", []))
    names |= set(project.get("optional-dependencies", {}).get(extra, []))
    return {re.match(r"[A-Za-z0-9._-]+", n.strip()).group(0) for n in names if n.strip()}


class TestDeclaredDependencies(unittest.TestCase):
    """The glmhmm extra must cover what the subpackage imports, at pinned versions."""

    def test_every_import_is_declared(self):
        """No module is imported that a clean install would not provide."""
        declared = _declared()
        missing = {
            name for name in _imported_top_level_modules(PACKAGE)
            if ALIASES.get(name, name) not in declared
        }
        self.assertEqual(missing, set(), f"imported but not declared: {sorted(missing)}")

    def test_dynamax_and_tfp_are_pinned_exactly(self):
        """dynamax leaves tfp-nightly unpinned, and only some nightlies work with jax 0.6.1."""
        try:
            import tomllib
        except ImportError:  # pragma: no cover - Python < 3.11
            raise unittest.SkipTest("needs tomllib")
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        extra = project["optional-dependencies"]["glmhmm"]
        for name in ("dynamax", "tfp-nightly"):
            pins = [r for r in extra if re.match(rf"{name}\s*==", r)]
            self.assertEqual(len(pins), 1, f"{name} must be pinned with == in [glmhmm]")


if __name__ == "__main__":
    unittest.main()
