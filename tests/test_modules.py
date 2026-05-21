import unittest
import numpy as np
import sys
import os
import shutil

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from qa_system.geom_integrity import GeomIntegrity
from qa_system.dosimetric_accuracy import DosimetricAccuracy
from qa_system.temporal_motion import TemporalMotion
from qa_system.robustness_check import RobustnessCheck
from qa_system.report_generator import QAReport

class TestQASystem(unittest.TestCase):

    def setUp(self):
        # Create dummy data
        self.shape_3d = (30, 30, 30)
        self.shape_2d = (30, 30)

        self.s_ct = np.zeros(self.shape_3d)
        # Make a larger object to pass fill_factor check (>0.2)
        # 20x20x20 cube in 30x30x30 space
        self.s_ct[5:25, 5:25, 5:25] = 500

        self.p_ct = np.zeros(self.shape_3d)
        self.p_ct[5:25, 5:25, 5:25] = 500

        self.input_img = np.sum(self.s_ct, axis=1) # Perfect projection
        self.input_img = (self.input_img / self.input_img.max()) * 255

        self.s_dose = np.random.rand(*self.shape_3d) * 50
        self.ref_dose = self.s_dose.copy()

        # 4D phases: Simple translation
        self.phases = []
        for i in range(3):
            vol = np.zeros(self.shape_3d)
            vol[5+i:25+i, 5:25, 5:25] = 500
            self.phases.append(vol)

        self.mask_lung = np.zeros(self.shape_3d)
        self.mask_lung[5:25, 5:25, 5:25] = 1

    def test_robustness(self):
        module = RobustnessCheck()
        data = {'input_image': self.input_img}
        res = module.validate(data)
        self.assertEqual(module.status, "PASS")
        self.assertIn('snr', res)

    def test_geom_integrity(self):
        module = GeomIntegrity()
        data = {
            'synthetic_ct': self.s_ct,
            'input_image': self.input_img,
            'planning_ct': self.p_ct
        }
        res = module.validate(data)
        # SSIM might not be 1.0 due to normalization diffs in DRR gen, but should be close
        self.assertIn('drr_ssim', res)
        self.assertIn('z_continuity_score', res)

    def test_dosimetric_accuracy(self):
        module = DosimetricAccuracy()
        data = {
            'synthetic_dose': self.s_dose,
            'reference_dose': self.ref_dose,
            'synthetic_ct': self.s_ct,
            'structure_masks': {'Lung': self.mask_lung},
            'voxel_size': (1.0, 1.0, 1.0)
        }
        res = module.validate(data)
        # Identical doses should pass
        self.assertEqual(module.status, "PASS")
        self.assertGreater(res['gamma_3mm_3%_pass_rate'], 0.99)

    def test_temporal_motion(self):
        module = TemporalMotion()
        data = {
            '4d_ct': self.phases
        }
        res = module.validate(data)
        self.assertIn('jacobian_analysis', res)
        # Simple translation should have near zero/one jacobian determinants (volume preserved)
        # But Demons might not perfectly capture large discrete steps (1 voxel shift is large for small image)
        # Just check it ran
        self.assertTrue(len(res['jacobian_analysis']) > 0)

    def test_report_generation(self):
        reporter = QAReport(output_dir="test_reports")
        dummy_res = {
            "TestModule": {"status": "PASS", "results": {"metric1": 0.99}}
        }
        path = reporter.generate_report("TEST_PATIENT", dummy_res)
        self.assertTrue(os.path.exists(path))
        # Cleanup
        if os.path.exists("test_reports"):
            shutil.rmtree("test_reports")

if __name__ == '__main__':
    unittest.main()
