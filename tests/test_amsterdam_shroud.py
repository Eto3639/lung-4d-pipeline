import numpy as np

from dvf_qa.amsterdam_shroud import (
    adaptive_search_band,
    build_shroud,
    respiratory_signal,
    respiratory_signal_adaptive,
    track_diaphragm,
)


def _synthetic_diaphragm_frames(
    n_frames: int = 120,
    height: int = 200,
    width: int = 160,
    breathing_hz: float = 0.25,
    fps: float = 15.0,
    amplitude_px: float = 25.0,
    base_row: float = 150.0,
    noise: float = 0.01,
    seed: int = 0,
):
    t = np.arange(n_frames) / fps
    diaphragm_y = base_row + amplitude_px * np.sin(2 * np.pi * breathing_hz * t)
    frames = np.zeros((n_frames, height, width), dtype=np.float32)
    y = np.arange(height).reshape(-1, 1)
    for i, dy in enumerate(diaphragm_y):
        frames[i] = 1.0 / (1.0 + np.exp(-(y - dy) / 1.5))
    rng = np.random.default_rng(seed)
    frames += rng.normal(0, noise, size=frames.shape).astype(np.float32)
    return frames, diaphragm_y


def test_build_shroud_shape():
    frames = np.zeros((30, 100, 80), dtype=np.float32)
    shroud = build_shroud(frames)
    assert shroud.shape == (99, 30)


def test_track_diaphragm_constant_peak():
    shroud = np.zeros((50, 20), dtype=np.float32)
    shroud[25, :] = 1.0
    rows = track_diaphragm(shroud, smoothing_sigma=0.0, search_band=(0.0, 1.0))
    assert rows.shape == (20,)
    assert np.allclose(rows, 25.0)


def test_recovers_breathing_frequency():
    fps = 15.0
    breathing_hz = 0.25
    frames, _ = _synthetic_diaphragm_frames(fps=fps, breathing_hz=breathing_hz)
    result = respiratory_signal(frames, fps=fps)
    assert abs(result.dominant_frequency_hz - breathing_hz) < 0.05


def test_tracked_position_correlates_with_truth():
    fps = 15.0
    frames, truth = _synthetic_diaphragm_frames(fps=fps, breathing_hz=0.25)
    result = respiratory_signal(frames, fps=fps)
    tracked = result.diaphragm_row - np.mean(result.diaphragm_row)
    truth_centered = truth - np.mean(truth)
    corr = np.corrcoef(tracked, truth_centered)[0, 1]
    assert abs(corr) > 0.95


def test_bandpass_suppresses_cardiac_component():
    fps = 15.0
    breathing_hz = 0.25
    cardiac_hz = 1.1
    t = np.arange(150) / fps
    diaphragm_y = (
        150.0
        + 25.0 * np.sin(2 * np.pi * breathing_hz * t)
        + 4.0 * np.sin(2 * np.pi * cardiac_hz * t)
    )
    frames = np.zeros((150, 200, 160), dtype=np.float32)
    y = np.arange(200).reshape(-1, 1)
    for i, dy in enumerate(diaphragm_y):
        frames[i] = 1.0 / (1.0 + np.exp(-(y - dy) / 1.5))
    result = respiratory_signal(frames, fps=fps, respiratory_band=(0.1, 0.6))
    assert abs(result.dominant_frequency_hz - breathing_hz) < 0.05


def test_adaptive_search_band_locks_onto_moving_row():
    """Synthetic shroud where only rows 40-60 have respiratory oscillation."""
    fps = 15.0
    T = 120
    Z = 100
    breathing_hz = 0.25
    t = np.arange(T) / fps
    osc = np.sin(2 * np.pi * breathing_hz * t)
    rng = np.random.default_rng(0)
    shroud = rng.normal(0, 0.05, size=(Z, T)).astype(np.float64)
    for z in range(40, 61):
        shroud[z] += osc
    lo, hi = adaptive_search_band(shroud, fps=fps,
                                   respiratory_band=(0.15, 0.6),
                                   select_quantile=0.7,
                                   fallback_band=(0.3, 1.0),
                                   min_peak_ratio=3.0)
    assert 0.3 <= lo <= 0.45
    assert 0.55 <= hi <= 0.7


def test_adaptive_search_band_falls_back_when_flat():
    """No structured oscillation → fallback band returned."""
    fps = 15.0
    rng = np.random.default_rng(1)
    shroud = rng.normal(0, 0.05, size=(100, 120))
    fallback = (0.4, 0.9)
    band = adaptive_search_band(shroud, fps=fps,
                                 fallback_band=fallback,
                                 min_peak_ratio=3.0)
    assert band == fallback


def test_respiratory_signal_adaptive_recovers_breathing_when_band_misaligned():
    """Phantom diaphragm at row ~50 of a 200-tall frame.

    Default fixed search_band (0.5, 0.95) covers rows 100-190 → diaphragm
    OUTSIDE. The adaptive variant should detect the row where the signal
    actually lives and lock onto it.
    """
    fps = 15.0
    breathing_hz = 0.25
    t = np.arange(150) / fps
    base_row = 50.0  # high in the frame
    diaphragm_y = base_row + 20.0 * np.sin(2 * np.pi * breathing_hz * t)
    frames = np.zeros((150, 200, 160), dtype=np.float32)
    y = np.arange(200).reshape(-1, 1)
    for i, dy in enumerate(diaphragm_y):
        frames[i] = 1.0 / (1.0 + np.exp(-(y - dy) / 1.5))
    rng = np.random.default_rng(0)
    frames += rng.normal(0, 0.01, size=frames.shape).astype(np.float32)

    fixed = respiratory_signal(frames, fps=fps)         # search_band (0.5, 0.95)
    adaptive, band = respiratory_signal_adaptive(frames, fps=fps, return_band=True)
    # Adaptive should pick a band that includes the high-up diaphragm
    assert band[0] < 0.5
    # Tracked diaphragm should correlate with the truth for adaptive,
    # while the fixed-band tracker tracks something else.
    truth = diaphragm_y - np.mean(diaphragm_y)
    adp_corr = abs(np.corrcoef(adaptive.diaphragm_row - np.mean(adaptive.diaphragm_row), truth)[0, 1])
    fix_corr = abs(np.corrcoef(fixed.diaphragm_row - np.mean(fixed.diaphragm_row), truth)[0, 1])
    assert adp_corr > 0.9
    assert adp_corr > fix_corr
