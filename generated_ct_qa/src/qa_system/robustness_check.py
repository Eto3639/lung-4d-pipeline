import numpy as np
from skimage import img_as_ubyte
from skimage.filters import sobel
from skimage.color import rgb2gray
from .base import QAModule

class RobustnessCheck(QAModule):
    def __init__(self):
        super().__init__("RobustnessCheck")

    def validate(self, data: dict) -> dict:
        """
        Validates basic image quality and FOV integrity.
        Input data expected keys:
        - 'input_image': 2D numpy array (X-ray/Fluoroscopy)
        """
        image = data.get('input_image')
        if image is None:
            return {"status": "ERROR", "message": "No input image provided"}

        # 1. SNR Check (Simple implementation)
        snr_threshold = self.config.get('snr_threshold', 2.0)
        snr_val, snr_status = self._check_snr(image, snr_threshold)
        self.results['snr'] = snr_val
        self.results['snr_status'] = snr_status

        # 2. FOV/Masking Check
        fov_status = self._check_fov(image)
        self.results['fov_status'] = fov_status

        # Overall Status
        if snr_status == "PASS" and fov_status == "PASS":
            self.status = "PASS"
        else:
            self.status = "FAIL"

        return self.results

    def _check_snr(self, image: np.ndarray, threshold: float = 2.0):
        """Calculates Signal-to-Noise Ratio."""
        # Improved Heuristic: Contrast / Noise
        # Assume background is in corners (common in X-ray/CT)
        h, w = image.shape
        corners = [
            image[0:h//10, 0:w//10],
            image[0:h//10, -w//10:],
            image[-h//10:, 0:w//10],
            image[-h//10:, -w//10:]
        ]
        background = np.concatenate([c.flatten() for c in corners])
        bg_mean = np.mean(background)
        bg_noise = np.std(background)

        # Center signal
        center = image[h//3:2*h//3, w//3:2*w//3]
        sig_mean = np.mean(center)

        # If background is perfectly clean (synthetic), noise is 0
        if bg_noise < 1e-6:
             return float('inf'), "PASS"

        snr = (sig_mean - bg_mean) / bg_noise
        status = "PASS" if snr > threshold else "FAIL"
        return snr, status

    def _check_fov(self, image: np.ndarray):
        """Checks if the image has significant blocking or cuts."""
        # Check for empty (black) borders which might indicate bad collimation or cropping
        # Threshold for "black"
        threshold = np.max(image) * 0.05

        h, w = image.shape
        # Check center region - should not be empty
        center_crop = image[h//4:3*h//4, w//4:3*w//4]
        if np.mean(center_crop) < threshold:
             # Center is dark -> Obstruction or empty
             print(f"FOV FAIL: Center mean {np.mean(center_crop)} < Threshold {threshold}")
             return "FAIL"

        # Check edges - this is highly specific to the X-ray machine geometry
        # For now, we assume if the image is mostly non-zero, it's fine.
        non_zeros = np.count_nonzero(image > threshold)
        fill_factor = non_zeros / image.size

        min_fill_factor = self.config.get('fov_fill_factor', 0.2)
        if fill_factor < min_fill_factor: # Too much empty space
            print(f"FOV FAIL: Fill factor {fill_factor} < {min_fill_factor}")
            return "FAIL"

        return "PASS"
