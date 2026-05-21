import sys
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qa_system.geom_integrity import GeomIntegrity
from qa_system.dosimetric_accuracy import DosimetricAccuracy
from qa_system.temporal_motion import TemporalMotion
from qa_system.robustness_check import RobustnessCheck
from qa_system.report_generator import QAReport

# Helper function for parallel execution (must be top-level)
def run_module(module_class, data):
    module = module_class()
    print(f"Starting {module.name}...")
    start_time = time.time()
    results = module.validate(data)
    duration = time.time() - start_time
    print(f"Finished {module.name} in {duration:.2f}s")
    return module.name, {"status": module.status, "results": results}

class QAManager:
    def __init__(self):
        self.modules = [
            RobustnessCheck,
            GeomIntegrity,
            DosimetricAccuracy,
            TemporalMotion
        ]
        self.reporter = QAReport()

    def process_patient(self, patient_id, data):
        """
        Runs all QA modules in parallel.
        """
        print(f"Processing Patient {patient_id} with {len(self.modules)} modules...")

        qa_results = {}

        # Parallel Execution Strategy
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            # Map futures to modules
            future_to_module = {
                executor.submit(run_module, mod_cls, data): mod_cls
                for mod_cls in self.modules
            }

            for future in as_completed(future_to_module):
                try:
                    name, res = future.result()
                    qa_results[name] = res
                except Exception as exc:
                    mod_name = future_to_module[future].__name__
                    print(f"{mod_name} generated an exception: {exc}")
                    qa_results[mod_name] = {"status": "ERROR", "error": str(exc)}

        # Generate Report
        report_path = self.reporter.generate_report(patient_id, qa_results)
        print(f"QA Completed. Report saved to: {report_path}")
        return qa_results

def generate_dummy_data():
    """Generates synthetic data for testing the pipeline."""
    shape_3d = (50, 50, 50)
    shape_2d = (50, 50)

    # 3D Volumes
    s_ct = np.random.rand(*shape_3d) * 1000 - 500 # HU range
    p_ct = np.random.rand(*shape_3d) * 1000 - 500

    # Doses
    s_dose = np.random.rand(*shape_3d) * 50 # Gy
    ref_dose = s_dose + (np.random.rand(*shape_3d) - 0.5) # Slight diff

    # 2D Image
    input_img = np.random.rand(*shape_2d) * 255

    # Masks
    mask_lung = np.zeros(shape_3d)
    mask_lung[10:40, 10:40, 10:40] = 1

    # 4D CT (List of 3 phases)
    phases = [s_ct + i*10 for i in range(3)]

    data = {
        "synthetic_ct": s_ct,
        "planning_ct": p_ct,
        "input_image": input_img,
        "synthetic_dose": s_dose,
        "reference_dose": ref_dose,
        "structure_masks": {"Lung": mask_lung},
        "4d_ct": phases,
        "voxel_size": (1.0, 1.0, 1.0)
    }
    return data

def main():
    manager = QAManager()
    data = generate_dummy_data()
    manager.process_patient("TEST_PATIENT_001", data)

if __name__ == "__main__":
    main()
