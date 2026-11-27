import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tetra.runtime.llm import read_api_key


class ApiKeyConfigurationTests(unittest.TestCase):
    def test_environment_and_explicit_file_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "credential"
            key_file.write_text("file-value\n")
            with patch.dict(os.environ, {"EXAMPLE_API_KEY_FILE": str(key_file)}, clear=True):
                self.assertEqual(read_api_key("EXAMPLE_API_KEY"), "file-value")
                os.environ["EXAMPLE_API_KEY"] = "environment-value"
                self.assertEqual(read_api_key("EXAMPLE_API_KEY", key_file), "environment-value")

    def test_missing_configuration_gives_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "EXAMPLE_API_KEY_FILE"):
                read_api_key("EXAMPLE_API_KEY")
