import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse
from scipy.ndimage import center_of_mass
from .base import QAModule

class GeomIntegrity(QAModule):
    def __init__(self):
        super().__init__("GeomIntegrity")

    def validate(self, data: dict) -> dict:
        """
        Validates Geometric Integrity.
        Input data:
        - 'synthetic_ct': 3D numpy array (Synthesized CT)
        - 'input_image': 2D numpy array (Original X-ray/Fluoroscopy) - Reference for DRR
        - 'planning_ct': 3D numpy array (Original Planning CT) - Reference for Anatomy
        """
        s_ct = data.get('synthetic_ct')
        input_img = data.get('input_image')
        p_ct = data.get('planning_ct')

        if s_ct is None:
            return {"status": "ERROR", "message": "No Synthetic CT provided"}

        # 1. DRR Reprojection & Comparison
        if input_img is not None:
            drr = self._generate_drr(s_ct)
            # Resize if necessary - assuming s_ct projection matches input_img geometry for now
            # In real world, geometry calibration is needed here.

            # Simple check: if shapes mismatch, we can't calculate SSIM directly without regridding
            # We will assume they are registered for this QA module scope.
            if drr.shape == input_img.shape:
                ssim_val = ssim(input_img, drr, data_range=drr.max() - drr.min())
                ncc_val = self._calculate_ncc(input_img, drr)
                self.results['drr_ssim'] = ssim_val
                self.results['drr_ncc'] = ncc_val

                # Generate Overlay Plot
                overlay_path = self._generate_overlay_plot(input_img, drr)
                self.results['plot_overlay_path'] = overlay_path
            else:
                 self.results['drr_status'] = "SKIPPED (Shape Mismatch)"

        # 2. Z-axis Continuity
        z_score, z_issues = self._analyze_z_continuity(s_ct)
        self.results['z_continuity_score'] = z_score
        self.results['z_issues'] = z_issues

        # 3. Anatomical Landmarks (Centroids)
        if p_ct is not None:
            landmarks = self._compare_landmarks(s_ct, p_ct)
            self.results['landmark_deviations'] = landmarks

        # Determine Status based on Config
        status = "PASS"
        if self.results.get('drr_ssim', 1.0) < self.config.get('drr_ssim_min', 0.8):
            status = "FAIL"
        if self.results.get('z_continuity_score', 1.0) < self.config.get('z_continuity_score_min', 0.8):
            status = "FAIL"

        self.status = status
        return self.results

    def _generate_drr(self, ct_volume: np.ndarray) -> np.ndarray:
        """Generates a pseudo-DRR by summing along the AP axis (assumed axis 1 or 0 depending on orientation)."""
        # Assuming axis 0 is Z (slices), axis 1 is Y (AP), axis 2 is X (LR)
        # We project along Y (AP view)
        drr = np.sum(ct_volume, axis=1)

        # Normalize to 0-255 or fit input range
        drr = (drr - drr.min()) / (drr.max() - drr.min() + 1e-6)
        return drr

    def _calculate_ncc(self, img1, img2):
        """Normalized Cross Correlation."""
        img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-6)
        img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-6)
        return np.mean(img1 * img2)

    def _analyze_z_continuity(self, ct_volume: np.ndarray):
        """Checks for sudden jumps in mean intensity along Z-axis."""
        # Calculate mean intensity per slice
        slice_means = np.mean(ct_volume, axis=(1, 2))
        gradients = np.abs(np.diff(slice_means))

        # Threshold for "Sudden Jump"
        threshold = np.mean(gradients) * 3 + 10 # Heuristic
        issues = np.where(gradients > threshold)[0].tolist()

        score = 1.0 - (len(issues) / len(slice_means))
        return score, issues

    def _compare_landmarks(self, s_ct, p_ct):
        """Compares centroids of Lung (Air) and Bone."""
        # Thresholds (HU)
        AIR_THRESH = -400
        BONE_THRESH = 400

        def get_centroids(vol):
            air_mask = vol < AIR_THRESH
            bone_mask = vol > BONE_THRESH

            # Handle empty masks
            if np.any(air_mask):
                c_air = center_of_mass(air_mask)
            else:
                c_air = (0,0,0)

            if np.any(bone_mask):
                c_bone = center_of_mass(bone_mask)
            else:
                c_bone = (0,0,0)

            return np.array(c_air), np.array(c_bone)

        s_air, s_bone = get_centroids(s_ct)
        p_air, p_bone = get_centroids(p_ct)

        diff_air = np.linalg.norm(s_air - p_air)
        diff_bone = np.linalg.norm(s_bone - p_bone)

        return {
            "lung_centroid_diff_mm": diff_air, # Assuming 1 voxel = 1 mm for simplicity
            "bone_centroid_diff_mm": diff_bone
        }

    def _generate_overlay_plot(self, img1, img2):
        """Generates an overlay of Input and DRR."""
        filename = f"/tmp/overlay_{uuid.uuid4().hex}.png"

        plt.figure(figsize=(6, 6))
        # Red channel: img1, Green channel: img2
        norm1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-6)
        norm2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6)

        rgb = np.zeros((*img1.shape, 3))
        rgb[..., 0] = norm1 # Red
        rgb[..., 1] = norm2 # Green
        rgb[..., 2] = norm1 * 0.5 + norm2 * 0.5 # Blue mix

        plt.imshow(rgb)
        plt.title("Overlay: Red=Input, Green=DRR")
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        return filename
