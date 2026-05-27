"""Amsterdam Shroud respiratory signal extraction.

Classical, AI-free respiratory motion estimator for a sequence of 2D thoracic
X-ray frames. Based on Zijp, Sonke, van Herk (2004): "Extraction of the
respiratory signal from sequential thorax cone-beam X-ray images".

Pipeline
--------
1. Cranio-caudal (vertical) image gradient per frame.
2. Sum the absolute gradient along the lateral axis -> 1D profile per frame.
3. Stack profiles in time -> 2D shroud image (rows = CC position, cols = time).
4. Track the diaphragm row per time column (sub-pixel parabolic fit on the
   per-column argmax of the gradient response within a configurable row band).
5. Optional zero-phase bandpass in the respiratory frequency range to suppress
   cardiac and slow drift components.

No lung mask, no learned weights. Works the same way for AP and lateral views
as long as the vertical image axis is cranio-caudal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class ShroudResult:
    shroud: np.ndarray
    diaphragm_row: np.ndarray
    signal: np.ndarray
    row_amplitude: np.ndarray
    dominant_frequency_hz: float
    period_frames: float


def build_shroud(frames: np.ndarray, *, lateral_axis: int = 2) -> np.ndarray:
    """Build the Amsterdam Shroud image from a (T, H, W) frame stack.

    The returned array has shape (H-1, T): rows are cranio-caudal position,
    columns are time. The diaphragm appears as a wavy bright ridge.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected frames with shape (T, H, W), got {frames.shape}")
    work = frames.astype(np.float32)
    gy = np.diff(work, axis=1)
    profile = np.abs(gy).sum(axis=lateral_axis)
    return profile.T


def track_diaphragm(
    shroud: np.ndarray,
    *,
    smoothing_sigma: float = 1.0,
    search_band: tuple[float, float] = (0.5, 0.95),
) -> np.ndarray:
    """Track the diaphragm row per time column with sub-pixel accuracy.

    ``search_band`` restricts the row range as a fraction of the shroud
    height; the diaphragm typically sits in the lower portion of the frame.
    """
    if shroud.ndim != 2:
        raise ValueError(f"Expected shroud with shape (H, T), got {shroud.shape}")
    height, n_frames = shroud.shape
    work = shroud
    if smoothing_sigma > 0:
        work = ndi.gaussian_filter(shroud, sigma=(smoothing_sigma, 0))

    lo = int(np.clip(search_band[0] * height, 0, height - 1))
    hi = int(np.clip(search_band[1] * height, lo + 1, height))
    window = work[lo:hi]

    argmax = window.argmax(axis=0)
    rows = np.empty(n_frames, dtype=np.float64)
    for t in range(n_frames):
        r = int(argmax[t])
        if 0 < r < window.shape[0] - 1:
            ym, y0, yp = window[r - 1, t], window[r, t], window[r + 1, t]
            denom = ym - 2.0 * y0 + yp
            offset = 0.5 * (ym - yp) / denom if denom != 0 else 0.0
        else:
            offset = 0.0
        rows[t] = lo + r + offset
    return rows


def respiratory_signal(
    frames: np.ndarray,
    *,
    fps: float,
    respiratory_band: tuple[float, float] = (0.1, 0.6),
    search_band: tuple[float, float] = (0.5, 0.95),
    shroud_smoothing_sigma: float = 1.0,
    detrend_order: int = 2,
) -> ShroudResult:
    """Extract a respiratory signal from a frame sequence via Amsterdam Shroud.

    The returned ``signal`` is the bandpassed diaphragm trace (zero-mean).
    ``row_amplitude`` gives the per-row activity in the respiratory band and
    is useful for diagnostics (e.g. plotting which CC region drives motion).
    ``detrend_order`` controls the polynomial removed from the diaphragm
    trace before bandpassing (0 = mean removal only, 1 = linear, 2 = quadratic);
    a quadratic detrend suppresses slow drifts that would otherwise dominate
    the FFT-based dominant-frequency estimate.
    """
    shroud = build_shroud(frames)
    diaphragm = track_diaphragm(
        shroud,
        smoothing_sigma=shroud_smoothing_sigma,
        search_band=search_band,
    )
    diaphragm_detrended = _polynomial_detrend(diaphragm, order=detrend_order)

    nyquist_ok = fps > 2.0 * respiratory_band[1]
    enough_samples = diaphragm.size >= 16
    if nyquist_ok and enough_samples:
        sos = butter(3, respiratory_band, btype="bandpass", fs=fps, output="sos")
        signal = sosfiltfilt(sos, diaphragm_detrended)
        bp_rows = sosfiltfilt(sos, shroud - shroud.mean(axis=1, keepdims=True), axis=1)
        row_amplitude = bp_rows.std(axis=1)
    else:
        signal = diaphragm_detrended
        row_amplitude = shroud.std(axis=1)

    dominant_freq, period_frames = _dominant_frequency(signal, fps)

    return ShroudResult(
        shroud=shroud,
        diaphragm_row=diaphragm,
        signal=signal,
        row_amplitude=row_amplitude,
        dominant_frequency_hz=dominant_freq,
        period_frames=period_frames,
    )


def save_shroud_quicklook(
    result: ShroudResult,
    path: str | Path,
    *,
    mean_frame: np.ndarray | None = None,
    fps: float | None = None,
) -> None:
    """Render a 3-panel QC PNG: optional mean frame, shroud + tracked line, signal."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = 2 + (1 if mean_frame is not None else 0)
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4), squeeze=False)
    axes = axes.ravel()
    idx = 0
    if mean_frame is not None:
        axes[idx].imshow(mean_frame, cmap="gray")
        axes[idx].set_title("Mean frame")
        axes[idx].axis("off")
        idx += 1

    axes[idx].imshow(result.shroud, aspect="auto", cmap="gray", origin="upper")
    axes[idx].plot(np.arange(result.diaphragm_row.size), result.diaphragm_row, color="tab:red", lw=1.0)
    axes[idx].set_title("Shroud + tracked diaphragm")
    axes[idx].set_xlabel("Frame")
    axes[idx].set_ylabel("CC position (px)")
    idx += 1

    t = np.arange(result.signal.size)
    if fps is not None and fps > 0:
        t = t / fps
        xlabel = "Time (s)"
    else:
        xlabel = "Frame"
    axes[idx].plot(t, result.signal, color="tab:blue", lw=1.0)
    axes[idx].set_title(
        f"Respiratory signal  |  f={result.dominant_frequency_hz:.3f} Hz"
        f"  (period {result.period_frames:.1f} frames)"
    )
    axes[idx].set_xlabel(xlabel)
    axes[idx].set_ylabel("Bandpassed CC offset (px)")
    axes[idx].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def adaptive_search_band(
    shroud: np.ndarray,
    fps: float,
    *,
    respiratory_band: tuple[float, float] = (0.15, 0.6),
    select_quantile: float = 0.7,
    fallback_band: tuple[float, float] = (0.3, 1.0),
    min_peak_ratio: float = 3.0,
) -> tuple[float, float]:
    """Auto-pick a ``search_band`` by scoring each shroud row's respiratory-band signal.

    For every CC row of the shroud, compute the standard deviation after a
    zero-phase band-pass filter restricted to the respiratory frequency range.
    Rows whose score exceeds the ``select_quantile`` quantile become the
    candidate diaphragm region; the band spans from min to max selected row
    (with light padding). When no row stands out (``max / median`` score below
    ``min_peak_ratio``) the function returns ``fallback_band`` so calling code
    can still produce a signal — useful when the diaphragm has fully left the
    field of view and the row scores are uniformly weak.
    """
    if shroud.ndim != 2:
        raise ValueError(f"expected shroud (Z, T); got {shroud.shape}")
    Z, T = shroud.shape
    if Z < 8 or T < 8:
        return fallback_band

    centered = shroud - shroud.mean(axis=1, keepdims=True)
    if fps > 2.0 * respiratory_band[1] and T >= 16:
        sos = butter(3, respiratory_band, btype="bandpass", fs=fps, output="sos")
        bp = sosfiltfilt(sos, centered, axis=1)
        scores = bp.std(axis=1)
    else:
        scores = centered.std(axis=1)

    median = float(np.median(scores))
    peak = float(np.max(scores))
    if median <= 0 or peak / max(median, 1e-9) < min_peak_ratio:
        return fallback_band

    threshold = float(np.quantile(scores, select_quantile))
    mask = scores >= threshold
    if mask.sum() < 4:
        return fallback_band

    # Keep only the largest contiguous run of high-score rows. Diaphragm
    # signal is anatomically contiguous along CC; isolated row spikes are
    # almost always noise that happens to pass the threshold.
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return fallback_band
    sizes = ndi.sum(mask, labels, index=np.arange(1, n_labels + 1))
    largest = int(np.argmax(sizes)) + 1
    selected = np.where(labels == largest)[0]

    lo, hi = int(selected.min()), int(selected.max())
    pad = max(1, (hi - lo) // 10)
    lo = max(0, lo - pad)
    hi = min(Z - 1, hi + pad)
    return (lo / Z, (hi + 1) / Z)


def respiratory_signal_adaptive(
    frames: np.ndarray,
    *,
    fps: float,
    respiratory_band: tuple[float, float] = (0.1, 0.6),
    shroud_smoothing_sigma: float = 1.0,
    detrend_order: int = 2,
    selection_band: tuple[float, float] = (0.15, 0.6),
    select_quantile: float = 0.7,
    fallback_band: tuple[float, float] = (0.3, 1.0),
    min_peak_ratio: float = 3.0,
    return_band: bool = False,
) -> ShroudResult | tuple[ShroudResult, tuple[float, float]]:
    """``respiratory_signal`` with an auto-chosen ``search_band``.

    Builds the shroud once, scores each row in the respiratory frequency
    band, picks the rows that carry the signal, and feeds that range back as
    ``search_band``. When ``return_band`` is True, also returns the selected
    ``(lo, hi)`` fractions for diagnostics.
    """
    shroud = build_shroud(frames)
    band = adaptive_search_band(
        shroud, fps,
        respiratory_band=selection_band,
        select_quantile=select_quantile,
        fallback_band=fallback_band,
        min_peak_ratio=min_peak_ratio,
    )
    result = respiratory_signal(
        frames,
        fps=fps,
        respiratory_band=respiratory_band,
        search_band=band,
        shroud_smoothing_sigma=shroud_smoothing_sigma,
        detrend_order=detrend_order,
    )
    if return_band:
        return result, band
    return result


def respiratory_signal_intensity_roi(
    frames: np.ndarray,
    *,
    fps: float,
    roi_band_y: tuple[float, float] = (0.45, 0.95),
    roi_band_x: tuple[float, float] = (0.25, 0.75),
    respiratory_band: tuple[float, float] = (0.1, 0.6),
) -> ShroudResult:
    """Alternative lateral-friendly signal: mean intensity of a central lower ROI.

    Useful when the Amsterdam Shroud diaphragm tracker locks onto a static
    high-gradient structure (spine, mediastinum) and produces a flat signal.
    The intensity in a fixed central-lower lung ROI tends to oscillate with
    air content (and thus respiration) without relying on edge tracking.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected frames (T, H, W); got {frames.shape}")
    T, H, W = frames.shape
    y0, y1 = int(roi_band_y[0] * H), max(int(roi_band_y[1] * H), int(roi_band_y[0] * H) + 1)
    x0, x1 = int(roi_band_x[0] * W), max(int(roi_band_x[1] * W), int(roi_band_x[0] * W) + 1)
    roi = frames[:, y0:y1, x0:x1].astype(np.float32)
    raw = roi.mean(axis=(1, 2))

    if fps > 2 * respiratory_band[1] and raw.size >= 16:
        sos = butter(3, respiratory_band, btype="bandpass", fs=fps, output="sos")
        signal = sosfiltfilt(sos, raw - raw.mean())
    else:
        signal = raw - raw.mean()
    freq, period = _dominant_frequency(signal, fps)

    shroud = roi.mean(axis=2).T
    return ShroudResult(
        shroud=shroud,
        diaphragm_row=np.full(T, (y0 + y1) / 2.0, dtype=np.float64),
        signal=signal,
        row_amplitude=shroud.std(axis=1),
        dominant_frequency_hz=freq,
        period_frames=period,
    )


def respiratory_signal_frame_diff(
    frames: np.ndarray,
    *,
    fps: float,
    crop_band_y: tuple[float, float] = (0.4, 0.95),
    respiratory_band: tuple[float, float] = (0.1, 0.6),
) -> ShroudResult:
    """Alternative signal: sum of absolute inter-frame intensity differences.

    Captures total motion in the lower thorax irrespective of which structure
    is moving. Robust when neither a clean diaphragm edge nor a stable
    ROI-intensity signal is available.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected frames (T, H, W); got {frames.shape}")
    T, H, W = frames.shape
    y0, y1 = int(crop_band_y[0] * H), max(int(crop_band_y[1] * H), int(crop_band_y[0] * H) + 1)
    cropped = frames[:, y0:y1, :].astype(np.float32)
    diff_mag = np.zeros(T, dtype=np.float64)
    diff_mag[1:] = np.abs(np.diff(cropped, axis=0)).sum(axis=(1, 2))
    diff_mag[0] = diff_mag[1] if T > 1 else 0.0

    if fps > 2 * respiratory_band[1] and diff_mag.size >= 16:
        sos = butter(3, respiratory_band, btype="bandpass", fs=fps, output="sos")
        signal = sosfiltfilt(sos, diff_mag - diff_mag.mean())
    else:
        signal = diff_mag - diff_mag.mean()
    freq, period = _dominant_frequency(signal, fps)

    shroud = cropped.mean(axis=2).T
    return ShroudResult(
        shroud=shroud,
        diaphragm_row=np.full(T, (y0 + y1) / 2.0, dtype=np.float64),
        signal=signal,
        row_amplitude=shroud.std(axis=1),
        dominant_frequency_hz=freq,
        period_frames=period,
    )


def _polynomial_detrend(signal: np.ndarray, *, order: int = 2) -> np.ndarray:
    """Subtract a fitted polynomial trend of given ``order`` from ``signal``.

    Falls back to mean removal when ``order <= 0`` or ``signal`` is too short.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if order <= 0 or arr.size < max(order + 1, 4):
        return arr - float(np.mean(arr))
    x = np.arange(arr.size, dtype=np.float64)
    coef = np.polyfit(x, arr, order)
    return arr - np.polyval(coef, x)


def _dominant_frequency(signal: np.ndarray, fps: float) -> tuple[float, float]:
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig - np.mean(sig)
    if sig.size < 4 or fps <= 0:
        return float("nan"), float("nan")
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fps)
    power = np.abs(spec) ** 2
    if power.size > 1:
        power[0] = 0.0
    peak = int(np.argmax(power))
    freq = float(freqs[peak])
    period = float(fps / freq) if freq > 0 else float("nan")
    return freq, period
