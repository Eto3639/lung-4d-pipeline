import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import uuid
from .base import QAModule

# Try importing pymedphys, or define a placeholder/simplified version
try:
    import pymedphys
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False

class DosimetricAccuracy(QAModule):
    def __init__(self):
        super().__init__("DosimetricAccuracy")

    def validate(self, data: dict) -> dict:
        """
        Validates Dosimetric Accuracy (HU, Gamma, DVH).
        Input data:
        - 'synthetic_dose': 3D numpy array (Dose on sCT)
        - 'reference_dose': 3D numpy array (Dose on pCT)
        - 'synthetic_ct': 3D numpy array (for HU check)
        - 'structure_masks': Dict of { 'PTV': mask, 'Lung': mask }
        """
        s_dose = data.get('synthetic_dose')
        ref_dose = data.get('reference_dose')
        s_ct = data.get('synthetic_ct')
        masks = data.get('structure_masks', {})

        if s_dose is None or ref_dose is None:
             return {"status": "ERROR", "message": "Dose distributions missing"}

        # 1. HU Profiling
        if s_ct is not None:
            self.results['hu_stats'] = self._check_hu_stats(s_ct, masks)

        # 2. Gamma Analysis
        # Voxel size is needed. Assuming 1mm isotropic for now or passed in metadata
        voxel_size = data.get('voxel_size', (1.0, 1.0, 1.0))

        gamma_33, pass_33 = self._calculate_gamma(ref_dose, s_dose, voxel_size, 3, 3)
        gamma_22, pass_22 = self._calculate_gamma(ref_dose, s_dose, voxel_size, 2, 2)

        self.results['gamma_3mm_3%_pass_rate'] = pass_33
        self.results['gamma_2mm_2%_pass_rate'] = pass_22

        # 3. DVH Parameters
        dvh_metrics, dvh_plot_path = self._calculate_dvh_metrics(s_dose, ref_dose, masks)
        self.results['dvh_metrics'] = dvh_metrics
        self.results['plot_dvh_path'] = dvh_plot_path

        # Status Logic
        threshold_33 = self.config.get('gamma_3mm_3%_pass_rate_min', 0.90)
        if pass_33 >= threshold_33:
            self.status = "PASS"
        else:
            self.status = "FAIL"

        return self.results

    def _check_hu_stats(self, ct, masks):
        stats = {}
        for name, mask in masks.items():
            if mask.shape != ct.shape:
                continue # Skip mismatch

            masked_voxels = ct[mask > 0]
            if len(masked_voxels) == 0:
                continue

            stats[name] = {
                "mean_hu": float(np.mean(masked_voxels)),
                "std_hu": float(np.std(masked_voxels))
            }
        return stats

    def _calculate_gamma(self, ref, eval_img, voxel_size, dta, dd):
        """
        Calculates Gamma Index.
        DTA: Distance to Agreement (mm)
        DD: Dose Difference (%)
        """
        if HAS_PYMEDPHYS:
            # Pymedphys implementation is highly optimized
            # We need coordinates. Creating dummy coords based on shape and voxel_size
            z, y, x = ref.shape
            coords = (
                np.arange(z) * voxel_size[0],
                np.arange(y) * voxel_size[1],
                np.arange(x) * voxel_size[2]
            )

            gamma = pymedphys.gamma(
                coords, ref,
                coords, eval_img,
                dose_percent_threshold=dd,
                distance_mm_threshold=dta,
                lower_percent_dose_cutoff=10, # Ignore low dose
                quiet=True
            )

            pass_rate = np.sum(gamma < 1) / np.sum(~np.isnan(gamma))
            return np.mean(gamma), pass_rate
        else:
            return self._calculate_gamma_numpy(ref, eval_img, voxel_size, dta, dd)

    def _calculate_gamma_numpy(self, reference_dose, eval_dose, voxel_size, dta_mm, dd_percent):
        """
        Calculates Gamma Index using pure NumPy/SciPy (Fallback).
        Uses a localized search window to improve performance over brute force.
        """
        # Thresholds
        max_dose = np.max(reference_dose)
        dose_threshold = max_dose * (dd_percent / 100.0) # Absolute dose diff threshold
        dist_threshold_sq = dta_mm ** 2

        # Normalize doses for gamma eq: DoseDiff / (DD%)
        # Here we compute the terms squared

        # Prepare result map
        gamma_map = np.full_like(reference_dose, np.inf)

        # Define search window in voxels
        # We need to search +/- ceil(dta / min_voxel_size)
        search_radius = [int(np.ceil(dta_mm / res)) for res in voxel_size]

        # Optimizing: Instead of looping over every reference voxel and searching,
        # we iterate over the search window offsets (convolution-like approach).
        # Shift the evaluation image and compute "partial gamma" for that shift.

        shifts = []
        for z in range(-search_radius[0], search_radius[0] + 1):
            for y in range(-search_radius[1], search_radius[1] + 1):
                for x in range(-search_radius[2], search_radius[2] + 1):
                    # Physical distance squared for this shift
                    dist_sq = (z * voxel_size[0])**2 + (y * voxel_size[1])**2 + (x * voxel_size[2])**2
                    if dist_sq > dist_threshold_sq:
                        continue
                    shifts.append((z, y, x, dist_sq))

        # For each shift, calculate Gamma candidate
        for dz, dy, dx, dist_sq in shifts:
            # Shifted Eval Image
            # Using scipy.ndimage.shift is slow; simpler array slicing is faster
            # shift > 0 means we look at eval_dose[z+dz], effectively comparing ref[z] to eval[z+dz]

            # Slice ranges
            # Ref: [max(0, -dz) : min(D, D-dz)]
            # Eval: [max(0, dz) : min(D, D+dz)]

            def get_slices(length, shift):
                src_start = max(0, shift)
                src_end = min(length, length + shift)
                dst_start = max(0, -shift)
                dst_end = min(length, length - shift)
                return slice(dst_start, dst_end), slice(src_start, src_end)

            sl_ref_z, sl_eval_z = get_slices(reference_dose.shape[0], dz)
            sl_ref_y, sl_eval_y = get_slices(reference_dose.shape[1], dy)
            sl_ref_x, sl_eval_x = get_slices(reference_dose.shape[2], dx)

            # Extract overlapping regions
            ref_chunk = reference_dose[sl_ref_z, sl_ref_y, sl_ref_x]
            eval_chunk = eval_dose[sl_eval_z, sl_eval_y, sl_eval_x]

            # Dose difference squared
            dose_diff_sq = (ref_chunk - eval_chunk) ** 2

            # Gamma squared for this specific neighbor relationship
            # Gamma^2 = (dist^2 / DTA^2) + (dose_diff^2 / DD^2)
            current_gamma_sq = (dist_sq / dist_threshold_sq) + (dose_diff_sq / (dose_threshold**2 + 1e-6))

            # Update minimum gamma found so far
            # We only update the region of gamma_map corresponding to sl_ref
            current_min = gamma_map[sl_ref_z, sl_ref_y, sl_ref_x]
            np.minimum(current_min, current_gamma_sq, out=current_min)

        # Finalize
        gamma_map = np.sqrt(gamma_map)

        # Statistics
        valid_voxels = np.sum(~np.isinf(gamma_map))
        passing_voxels = np.sum(gamma_map <= 1.0)

        pass_rate = passing_voxels / (valid_voxels + 1e-6)

        return float(np.mean(gamma_map[gamma_map < np.inf])), pass_rate

    def _calculate_dvh_metrics(self, s_dose, ref_dose, masks):
        metrics = {}
        plot_data = {}

        for name, mask in masks.items():
            if mask.shape != s_dose.shape:
                continue

            # Get voxels in structure
            s_voxels = s_dose[mask > 0]
            ref_voxels = ref_dose[mask > 0]

            if len(s_voxels) == 0:
                continue

            # Calculate DVH stats
            s_sorted = np.sort(s_voxels)[::-1]
            ref_sorted = np.sort(ref_voxels)[::-1]

            # D95
            def get_d95(sorted_voxels):
                idx = int(len(sorted_voxels) * 0.95)
                return sorted_voxels[idx]

            # V20
            def get_v20(voxels):
                return np.sum(voxels >= 20) / len(voxels) * 100

            metrics[name] = {
                "delta_D95": float(get_d95(s_sorted) - get_d95(ref_sorted)),
                "delta_V20": float(get_v20(s_voxels) - get_v20(ref_voxels))
            }

            plot_data[name] = (s_sorted, ref_sorted)

        # Generate Plot
        filename = f"/tmp/dvh_{uuid.uuid4().hex}.png"
        plt.figure(figsize=(8, 5))
        for name, (s_data, ref_data) in plot_data.items():
            # X axis: Dose, Y axis: Volume (%)
            # Simple cumulative histogram
            x_s = np.linspace(0, 100, len(s_data)) # Percent volume
            plt.plot(s_data, x_s, label=f"{name} Synthetic")

            x_ref = np.linspace(0, 100, len(ref_data))
            plt.plot(ref_data, x_ref, linestyle='--', label=f"{name} Reference")

        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume (%)")
        plt.title("DVH Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        return metrics, filename
