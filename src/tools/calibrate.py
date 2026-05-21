import sys
import os
import json
import numpy as np
from typing import List, Dict

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qa_system.geom_integrity import GeomIntegrity
from qa_system.dosimetric_accuracy import DosimetricAccuracy
from qa_system.temporal_motion import TemporalMotion
from qa_system.robustness_check import RobustnessCheck
from main import generate_dummy_data # Reuse dummy data for demo

def run_calibration(num_samples: int = 5):
    """
    Runs QA modules on a set of reference data and suggests thresholds.
    """
    print(f"Running calibration on {num_samples} samples...")

    # Storage for metrics
    metrics = {
        "RobustnessCheck": {"snr": [], "fov_fill_factor": []},
        "GeomIntegrity": {"drr_ssim": [], "z_continuity_score": []},
        "DosimetricAccuracy": {"gamma_3mm_3%_pass_rate": [], "gamma_2mm_2%_pass_rate": []},
        "TemporalMotion": {"smoothness_score": [], "negative_jacobian_fraction": []}
    }

    # 1. Collect Data
    for i in range(num_samples):
        data = generate_dummy_data()

        # Run Modules
        # Robustness
        r_mod = RobustnessCheck()
        res = r_mod.validate(data)
        metrics["RobustnessCheck"]["snr"].append(res.get("snr", 0))
        # Fill factor hack: we need to access internal logic or just rely on what's exposed.
        # Current RobustnessCheck doesn't expose fill_factor in results explicitly, only status.
        # We should update RobustnessCheck to return fill_factor if we want to calibrate it.
        # For now, we skip internal vars if not in results.

        # Geom
        g_mod = GeomIntegrity()
        res = g_mod.validate(data)
        metrics["GeomIntegrity"]["drr_ssim"].append(res.get("drr_ssim", 0))
        metrics["GeomIntegrity"]["z_continuity_score"].append(res.get("z_continuity_score", 0))

        # Dose
        d_mod = DosimetricAccuracy()
        res = d_mod.validate(data)
        metrics["DosimetricAccuracy"]["gamma_3mm_3%_pass_rate"].append(res.get("gamma_3mm_3%_pass_rate", 0))
        metrics["DosimetricAccuracy"]["gamma_2mm_2%_pass_rate"].append(res.get("gamma_2mm_2%_pass_rate", 0))

        # Temporal
        t_mod = TemporalMotion()
        res = t_mod.validate(data)
        metrics["TemporalMotion"]["smoothness_score"].append(res.get("trajectory_smoothness", 0))
        # Flatten jacobian stats
        jac_stats = res.get("jacobian_analysis", [])
        if jac_stats:
            neg_fracs = [j['negative_jacobian_fraction'] for j in jac_stats]
            metrics["TemporalMotion"]["negative_jacobian_fraction"].extend(neg_fracs)

    # 2. Calculate Statistics & Thresholds
    suggested_config = {}

    for module, metric_dict in metrics.items():
        suggested_config[module] = {}
        for metric_name, values in metric_dict.items():
            if not values:
                continue

            vals = np.array(values)
            # Filter infs/nans
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            mean = np.mean(vals)
            std = np.std(vals)

            # Logic:
            # If metric is "Higher is Better" (e.g., SSIM, Gamma Pass), threshold = Mean - 2*STD
            # If metric is "Lower is Better" (e.g., Negative Jacobian), threshold = Mean + 2*STD

            # Heuristic mapping
            lower_is_better = ["negative_jacobian_fraction"]

            if metric_name in lower_is_better:
                thresh = mean + 2 * std
                suggested_config[module][f"{metric_name}_max"] = float(thresh)
            else:
                thresh = mean - 2 * std
                # Clamp to 0-1 if it's a ratio
                if "rate" in metric_name or "score" in metric_name or "ssim" in metric_name:
                    thresh = max(0.0, min(1.0, thresh))
                suggested_config[module][f"{metric_name}_min"] = float(thresh)

    # 3. Output
    output_path = "config/suggested_thresholds.json"
    with open(output_path, 'w') as f:
        json.dump(suggested_config, f, indent=4)

    print(f"Calibration complete. Suggested thresholds saved to {output_path}")
    print(json.dumps(suggested_config, indent=4))

if __name__ == "__main__":
    run_calibration()
