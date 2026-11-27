import unittest
from pathlib import Path
from tetra.runtime.inputs import validate_config
from tetra.prediction.backend import command_for

class PotentialsTests(unittest.TestCase):
    def test_opt_in_changes_command_and_cache_configuration(self):
        config = validate_config({'use_potentials': True})
        self.assertIn('--use_potentials', command_for(Path('/tmp/example'), config, False))
        self.assertNotIn('--use_potentials', command_for(Path('/tmp/example'), validate_config(), False))
        self.assertNotEqual(config, validate_config())

    def test_truthy_string_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({'use_potentials': 'false'})
