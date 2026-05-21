import unittest
import os
import json
import numpy as np
import shutil
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from qa_system.config_manager import ConfigManager
from qa_system.dosimetric_accuracy import DosimetricAccuracy
from tools.calibrate import run_calibration

class TestConfigAndCalibration(unittest.TestCase):

    def setUp(self):
        # Backup existing config
        self.config_path = os.path.join(os.getcwd(), 'config', 'thresholds.json')
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, self.config_path + ".bak")

        # Create dummy data for testing
        self.shape_3d = (30, 30, 30)
        self.s_dose = np.random.rand(*self.shape_3d) * 50
        self.ref_dose = self.s_dose.copy() # Perfect match -> 100% pass
        self.voxel_size = (1.0, 1.0, 1.0)
        self.data = {
            'synthetic_dose': self.s_dose,
            'reference_dose': self.ref_dose,
            'voxel_size': self.voxel_size,
            'structure_masks': {}
        }

    def tearDown(self):
        # Restore config
        if os.path.exists(self.config_path + ".bak"):
            shutil.move(self.config_path + ".bak", self.config_path)
            # Force reload
            ConfigManager().reload()

    def test_dynamic_threshold_pass(self):
        # Set strict threshold that should PASS (1.0 vs 0.99)
        # Wait, perfect match is 1.0.
        cfg = {"DosimetricAccuracy": {"gamma_3mm_3%_pass_rate_min": 0.99}}
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f)

        ConfigManager().reload()

        module = DosimetricAccuracy()
        res = module.validate(self.data)

        self.assertEqual(module.status, "PASS", "Should pass with 100% rate vs 99% threshold")

    def test_dynamic_threshold_fail(self):
        # Set impossible threshold (1.01)
        cfg = {"DosimetricAccuracy": {"gamma_3mm_3%_pass_rate_min": 1.01}}
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f)

        ConfigManager().reload()

        module = DosimetricAccuracy()
        res = module.validate(self.data)

        self.assertEqual(module.status, "FAIL", "Should fail with 100% rate vs 101% threshold")

    def test_calibration_tool(self):
        # Run calibration
        run_calibration(num_samples=2)

        output_path = "config/suggested_thresholds.json"
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertIn("DosimetricAccuracy", data)
        self.assertIn("gamma_3mm_3%_pass_rate_min", data["DosimetricAccuracy"])

if __name__ == '__main__':
    unittest.main()
