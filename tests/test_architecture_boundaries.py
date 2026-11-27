"""Protect the research workflow's dependencies across package boundaries."""
import ast
from importlib.util import resolve_name
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / "tetra"


def imported_modules(path):
    package_name = ".".join(("tetra", *path.parent.relative_to(PACKAGE).parts))
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package_name)
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_online_packages_do_not_import_offline_benchmarks(self):
        for package in ("orchestration", "research", "prediction", "evaluation", "runtime"):
            for path in (PACKAGE / package).rglob("*.py"):
                with self.subTest(path=path.relative_to(PACKAGE)):
                    modules = imported_modules(path)
                    self.assertFalse(any(
                        module == "tetra.benchmarks"
                        or module.startswith("tetra.benchmarks.")
                        or module in {
                            "tetra.batch", "tetra.benchmark", "tetra.campaign",
                            "tetra.group_evaluation", "tetra.overnight_agent",
                            "tetra.research_benchmark",
                        }
                        for module in modules
                    ))

    def test_single_agent_does_not_import_team_runner_for_tool_schema(self):
        for filename in ("scientific_agent.py", "board_loop.py"):
            with self.subTest(filename=filename):
                modules = imported_modules(PACKAGE / "orchestration" / filename)
                self.assertFalse(any(
                    module.startswith(("tetra.orchestration.scientist_team", "tetra.scientist_team"))
                    for module in modules
                ))
