from __future__ import annotations
import pyabf
import numpy as np
import tensorflow as tf
import miniML

def init_containers(
        n_sweeps: int,
        event_len: int
) -> tuple[
    np.ndarray, np.ndarray,  # events, events_x
    np.ndarray, np.ndarray, np.ndarray,  # amplitudes, peak_values, peak_x
    np.ndarray, np.ndarray,  # bsl_start_x, bsl_end_x
    np.ndarray,              # scores
    np.ndarray, np.ndarray,  # tau, charge
    np.ndarray,              # risetimes
    np.ndarray, np.ndarray,  # rise_min_x, rise_max_x
    np.ndarray,              # slopes
    np.ndarray,              # decaytimes
    np.ndarray               # halfwidths
]:
    """
    Preallocate arrays for one-event-per-sweep results.
    Parameters
    ----------
    n_sweeps : int
        Number of sweeps in the recording.
    event_len : int
        Length of each event waveform.
    Returns
    -------
    tuple
        tuple of preallocated numpy arrays for storing event data and properties.
    """
    # per-sample waveforms/time
    events   = np.full((n_sweeps, event_len), np.nan, dtype=float)
    events_x = np.full((n_sweeps, event_len), np.nan, dtype=float)

    # per-sweep scalars
    amplitudes = np.full(n_sweeps, np.nan, dtype=float)
    peak_values = np.full(n_sweeps, np.nan, dtype=float)
    peak_x = np.full(n_sweeps, np.nan, dtype=float)
    bsl_start_x = np.full(n_sweeps, np.nan, dtype=float)
    bsl_end_x = np.full(n_sweeps, np.nan, dtype=float)
    scores = np.full(n_sweeps, np.nan, dtype=float)
    tau = np.full(n_sweeps, np.nan, dtype=float)
    charges = np.full(n_sweeps, np.nan, dtype=float)
    risetimes = np.full(n_sweeps, np.nan, dtype=float)
    rise_min_x = np.full(n_sweeps, np.nan, dtype=float)
    rise_max_x = np.full(n_sweeps, np.nan, dtype=float)
    slopes = np.full(n_sweeps, np.nan, dtype=float)
    decaytimes = np.full(n_sweeps, np.nan, dtype=float)
    halfwidths = np.full(n_sweeps, np.nan, dtype=float)

    return (events, events_x,
            amplitudes, peak_values, peak_x,
            bsl_start_x, bsl_end_x,
            scores, tau, charges, risetimes,
            rise_min_x, rise_max_x,
            slopes, decaytimes, halfwidths)


def _pad_or_crop_events(events: np.ndarray, event_len: int) -> np.ndarray:
    """Pad or crop event waveforms to a fixed length.

    Parameters
    ----------
    events
        Event waveform array with shape (n_events, n_samples).
    event_len
        Target number of samples per event.

    Returns
    -------
    np.ndarray
        Event waveforms with shape (n_events, event_len).

    """

    n_events = events.shape[0]
    out = np.full((n_events, event_len), np.nan, dtype=float)
    n_copy = min(events.shape[1], event_len)
    out[:, :n_copy] = events[:, :n_copy]
    return out


def _to_seconds(
        values: np.ndarray,
        sample_rate: float,
        trace_len: int
) -> np.ndarray:
    """
    Convert time-like values to seconds when values are sample indices.
    If the values are already in seconds, they are returned unchanged.

    Parameters
    ----------
    values
        Time-like values from miniML.
    sample_rate
        Acquisition sampling rate in Hz.
    trace_len
        Number of samples in the analyzed trace.

    Returns
    -------
    np.ndarray
        Values expressed in seconds.

    """

    if values.size == 0:
        return values

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return values

    trace_duration_s = trace_len / sample_rate
    max_abs = np.nanmax(np.abs(finite_values))
    
    if max_abs > (trace_duration_s * 1.5):
        return values / sample_rate
    
    return values


def _extract_sweep_events(
        detector: miniML.EventDetection,
        event_len: int,
        sample_rate: float
) -> dict[str, np.ndarray]:
    """
    Extract all detected events for one sweep.
    Based on the output of `miniML.EventDetection.detect_events`, extract all
    events for one sweep and compute associated properties.
    Stores waveforms and properties in a dictionary of numpy arrays, one entry per event.

    Parameters
    ----------
    detector
        miniML event detector after `detect_events`.
    event_len
        Target waveform length for output events.
    sample_rate
        Acquisition sampling rate in Hz.

    Returns
    -------
    dict[str, np.ndarray]
        Per-event arrays for one sweep.

    """

    raw_events = detector.events
    if raw_events is None or raw_events.shape[0] == 0:
        return {
            "events": np.empty((0, event_len), dtype=float),
            "events_x": np.empty((0, event_len), dtype=float),
            "amplitudes": np.empty(0, dtype=float),
            "peak_values": np.empty(0, dtype=float),
            "peak_x": np.empty(0, dtype=float),
            "bsl_start_x": np.empty(0, dtype=float),
            "bsl_end_x": np.empty(0, dtype=float),
            "scores": np.empty(0, dtype=float),
            "tau": np.empty(0, dtype=float),
            "charges": np.empty(0, dtype=float),
            "risetimes": np.empty(0, dtype=float),
            "rise_min_x": np.empty(0, dtype=float),
            "rise_max_x": np.empty(0, dtype=float),
            "slopes": np.empty(0, dtype=float),
            "decaytimes": np.empty(0, dtype=float),
            "halfwidths": np.empty(0, dtype=float)
        }

    # Pad or crop events to the target length
    events = _pad_or_crop_events(np.asarray(raw_events, dtype=float), event_len)
    n_events = events.shape[0]

    sampling = float(detector.trace.sampling)
    events_x = np.tile(np.arange(event_len, dtype=float) * sampling,
                       (n_events, 1))

    peak_values = np.asarray(detector.event_peak_values, dtype=float)
    event_bsls = np.asarray(detector.event_bsls, dtype=float)
    amplitudes = peak_values - event_bsls

    peak_x = np.asarray(detector.event_peak_times, dtype=float)
    peak_x = _to_seconds(peak_x, sample_rate, len(detector.trace.data))

    bsl_start_x = np.asarray(detector.bsl_starts, dtype=float) * sampling
    bsl_end_x = np.asarray(detector.bsl_ends, dtype=float) * sampling

    event_scores = getattr(detector, "event_scores", None)
    if event_scores is None:
        scores = np.full(n_events, np.nan, dtype=float)
    else:
        scores = np.asarray(event_scores, dtype=float)

    tau_values = getattr(detector, "taus", None)
    if tau_values is not None:
        tau = np.asarray(tau_values, dtype=float)
    elif detector.avg_decay_fit is not None:
        tau = np.full(n_events, float(detector.avg_decay_fit[1]), dtype=float)
    else:
        tau = np.full(n_events, np.nan, dtype=float)

    charges = np.asarray(getattr(detector, "charges", np.full(
        n_events, np.nan, dtype=float)), dtype=float)
    risetimes = np.asarray(detector.risetimes, dtype=float)
    rise_min_x = np.asarray(detector.min_positions_rise, dtype=float)
    rise_min_x *= sampling
    rise_max_x = np.asarray(detector.max_positions_rise, dtype=float)
    rise_max_x *= sampling
    slopes = np.asarray(getattr(detector, "slopes", np.full(
        n_events, np.nan, dtype=float)), dtype=float)
    decaytimes = np.asarray(getattr(detector, "decaytimes", np.full(
        n_events, np.nan, dtype=float)), dtype=float)
    halfwidths = np.asarray(getattr(detector, "halfwidths", np.full(
        n_events, np.nan, dtype=float)), dtype=float)

    return {
        "events": events,
        "events_x": events_x,
        "amplitudes": amplitudes,
        "peak_values": peak_values,
        "peak_x": peak_x,
        "bsl_start_x": bsl_start_x,
        "bsl_end_x": bsl_end_x,
        "scores": scores,
        "tau": tau,
        "charges": charges,
        "risetimes": risetimes,
        "rise_min_x": rise_min_x,
        "rise_max_x": rise_max_x,
        "slopes": slopes,
        "decaytimes": decaytimes,
        "halfwidths": halfwidths
    }


def _select_event_index(
        sweep_events: dict[str, np.ndarray],
        window_start_s: float,
        window_end_s: float
) -> int | None:
    """
    Select one event inside a time window for a sweep.
    The time window is defined by `window_start_s` and `window_end_s`
    in seconds from sweep start. If multiple events are in the window,
    the one with the largest absolute amplitude is selected.

    Parameters
    ----------
    sweep_events
        Per-event outputs for one sweep.
    window_start_s
        Start of evoked window in seconds from sweep start.
    window_end_s
        End of evoked window in seconds from sweep start.

    Returns
    -------
    int or None
        Index of selected event or None if no event is in the window.

    """

    peak_x = sweep_events["peak_x"]
    in_window = np.where(
        (peak_x >= window_start_s) &
        (peak_x <= window_end_s))[0]
    
    if in_window.size == 0:
        return None

    amplitudes = sweep_events["amplitudes"][in_window]
    best_local = int(np.nanargmax(np.abs(amplitudes)))
    
    return int(in_window[best_local])


def _subset_sweep_events(
        sweep_events: dict[str, np.ndarray],
        event_indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Subset one-sweep event arrays by indices.

    Parameters
    ----------
    sweep_events
        Per-event outputs for one sweep.
    event_indices
        Indices of events to keep.

    Returns
    -------
    dict[str, np.ndarray]
        Filtered per-event arrays.

    """

    return {
        key: value[event_indices]
        for key, value in sweep_events.items()
    }


def _find_events_in_windows(
        peak_x: np.ndarray,
        windows_s: list[tuple[float, float]]
) -> np.ndarray:
    """Find event indices inside any of the provided windows.

    Parameters
    ----------
    peak_x
        Event peak times in seconds from sweep start.
    windows_s
        Inclusive start/exclusive end windows in seconds.

    Returns
    -------
    np.ndarray
        Sorted unique event indices contained in any window.

    """

    if peak_x.size == 0:
        return np.empty(0, dtype=int)

    mask = np.zeros(peak_x.size, dtype=bool)
    for start_s, end_s in windows_s:
        if end_s <= start_s:
            continue
        mask |= (peak_x >= start_s) & (peak_x < end_s)

    return np.where(mask)[0]


def _get_evoked_window_duration_s(ephy_instance) -> float:
    """Return evoked window duration in seconds from imaging metadata."""

    dataset = ephy_instance._dataset
    n_frames = dataset[0].n_frames
    frame_rate = dataset[0].frame_rate
    return n_frames / frame_rate


def _extract_evoked_events_from_all_events(
        all_events: list[dict[str, np.ndarray]],
        abf: pyabf.ABF,
        event_len: int,
        window_start_s: float,
        window_end_s: float
) -> tuple:
    """Extract one evoked event per sweep from a precomputed all-events list."""

    n_sweeps = len(abf.sweepList)

    (events, events_x,
     amplitudes, peak_values, peak_x,
     bsl_start_x, bsl_end_x,
     scores, tau, charges,
     risetimes, rise_min_x, rise_max_x,
     slopes, decaytimes, halfwidths) = init_containers(n_sweeps, event_len)

    for sweep_n in abf.sweepList:
        sweep_events = all_events[sweep_n]
        idx = _select_event_index(sweep_events, window_start_s, window_end_s)

        if idx is None:
            continue

        events[sweep_n, :] = sweep_events["events"][idx]
        events_x[sweep_n, :] = sweep_events["events_x"][idx]
        amplitudes[sweep_n] = sweep_events["amplitudes"][idx]
        peak_values[sweep_n] = sweep_events["peak_values"][idx]
        peak_x[sweep_n] = sweep_events["peak_x"][idx]
        bsl_start_x[sweep_n] = sweep_events["bsl_start_x"][idx]
        bsl_end_x[sweep_n] = sweep_events["bsl_end_x"][idx]
        scores[sweep_n] = sweep_events["scores"][idx]
        tau[sweep_n] = sweep_events["tau"][idx]
        charges[sweep_n] = sweep_events["charges"][idx]
        risetimes[sweep_n] = sweep_events["risetimes"][idx]
        rise_min_x[sweep_n] = sweep_events["rise_min_x"][idx]
        rise_max_x[sweep_n] = sweep_events["rise_max_x"][idx]
        slopes[sweep_n] = sweep_events["slopes"][idx]
        decaytimes[sweep_n] = sweep_events["decaytimes"][idx]
        halfwidths[sweep_n] = sweep_events["halfwidths"][idx]

    return (
        events,
        events_x,
        amplitudes,
        peak_values,
        peak_x,
        bsl_start_x,
        bsl_end_x,
        scores,
        tau,
        charges,
        risetimes,
        rise_min_x,
        rise_max_x,
        slopes,
        decaytimes,
        halfwidths)


def BLA_EPSCs(
        ephy_instance,
        file_path,
        model: tf.keras.models.Model = None,
        channel: int = 0,
        scaling: int = 1,
        unit: str = 'pA',
        win_size: int = 3600,
        direction: str = 'negative',
        all_events: list[dict[str, np.ndarray]] = None,
        bla_epoch_idx: int = 5
) -> tuple:
    """Analyze BLA-evoked EPSCs from precomputed all-events."""

    abf = pyabf.ABF(file_path)
    add_points = int(win_size / 3)
    event_len = add_points + win_size + add_points

    if all_events is None:
        if model is None:
            raise ValueError("Provide either `model` or `all_events`.")

        all_events = detect_all_events(
            ephy_instance=ephy_instance,
            file_path=file_path,
            model=model,
            channel=channel,
            scaling=scaling,
            unit=unit,
            win_size=win_size,
            direction=direction)

    evoked_window_s = _get_evoked_window_duration_s(ephy_instance)
    t_start = abf.sweepEpochs.p1s[bla_epoch_idx] / abf.sampleRate
    t_end = t_start + evoked_window_s

    return _extract_evoked_events_from_all_events(
        all_events=all_events,
        abf=abf,
        event_len=event_len,
        window_start_s=t_start,
        window_end_s=t_end)


def CA3_EPSCs(
        ephy_instance,
        file_path,
        model: tf.keras.models.Model = None,
        channel: int = 0,
        scaling: int = 1,
        unit: str = 'pA',
        win_size: int = 3600,
        direction: str = 'negative',
        all_events: list[dict[str, np.ndarray]] = None,
        ca3_epoch_idx: int = 6
) -> tuple:
    """Analyze CA3-evoked EPSCs from precomputed all-events."""

    abf = pyabf.ABF(file_path)
    add_points = int(win_size / 3)
    event_len = add_points + win_size + add_points

    if all_events is None:
        if model is None:
            raise ValueError("Provide either `model` or `all_events`.")

        all_events = detect_all_events(
            ephy_instance=ephy_instance,
            file_path=file_path,
            model=model,
            channel=channel,
            scaling=scaling,
            unit=unit,
            win_size=win_size,
            direction=direction)

    evoked_window_s = _get_evoked_window_duration_s(ephy_instance)
    t_start = abf.sweepEpochs.p1s[ca3_epoch_idx] / abf.sampleRate
    t_end = t_start + evoked_window_s

    return _extract_evoked_events_from_all_events(
        all_events=all_events,
        abf=abf,
        event_len=event_len,
        window_start_s=t_start,
        window_end_s=t_end)


def detect_all_events(
        ephy_instance,
        file_path,
        model: tf.keras.models.Model,
        channel: int = 0,
        scaling: int = 1,
        unit: str = 'pA',
        win_size: int = 3600,
        direction: str = 'negative'
    ) -> list[dict[str, np.ndarray]]:
    """
    Detect all synaptic events in every sweep.
    
    Parameters
    ----------
    ephy_instance : Ephydataset
        Unused placeholder for API compatibility.
    file_path : str
        Path to the ABF file.
    channel : int, optional
        The channel index to analyze. Default is 0.
    scaling : int, optional
        Scaling factor for the data. Default is 1.
    unit : str, optional
        Unit of the data. Default is 'pA'.
    win_size : int, optional
        Window size for event detection. Default is 3600.
    direction : str, optional
        Direction of events to detect ('negative' or 'positive'). Default is 'negative'.
    Returns
    -------
    list of dict
        One dictionary per sweep with per-event arrays.
    """

    _ = ephy_instance

    abf = pyabf.ABF(file_path)

    add_points = int(win_size / 3)
    event_len = add_points + win_size + add_points
    all_events: list[dict[str, np.ndarray]] = []

    for sweep_n in abf.sweepList:
        abf.setSweep(sweep_n)

        # load from ABF file
        trace = miniML.MiniTrace.from_axon_file(
            filename=file_path,
            channel=channel,
            sweep=sweep_n,
            scaling=scaling,
            unit=unit)

        detector = miniML.EventDetection(
            data=trace,
            model=model,
            window_size=win_size,
            model_threshold=0.5,
            batch_size=512,
            event_direction=direction,
            verbose=0)

        detector.detect_events(
            eval=True,
            peak_w=5,
            rel_prom_cutoff=0.25,
            convolve_win=20,
            gradient_convolve_win=40,
            resample_to_600=True)

        sweep_events = _extract_sweep_events(
            detector=detector,
            event_len=event_len,
            sample_rate=abf.sampleRate)
        
        all_events.append(sweep_events)

    return all_events


def sEPSCs(
        ephy_instance,
        file_path,
        model: tf.keras.models.Model = None,
        channel: int = 0,
        scaling: int = 1,
        unit: str = 'pA',
        win_size: int = 3600,
        direction: str = 'negative',
        bla_epoch_idx: int = 5,
    ca3_epoch_idx: int = 6,
    all_events: list[dict[str, np.ndarray]] = None
) -> list[dict[str, np.ndarray]]:
    """Extract spontaneous EPSCs outside BLA and CA3 evoked windows.

    Spontaneous windows are defined as:
    1) from the end of the BLA-evoked window to the CA3 stimulus start,
    2) from the end of the CA3-evoked window to the end of the sweep.

    Parameters
    ----------
    ephy_instance : Ephydataset
        The Ephydataset instance containing imaging timing information.
    file_path : str
        Path to the ABF file.
    model : tf.keras.models.Model
        miniML model used for event detection.
    channel : int, optional
        Channel index to analyze, by default 0.
    scaling : int, optional
        Scaling factor for the trace, by default 1.
    unit : str, optional
        Signal unit label, by default 'pA'.
    win_size : int, optional
        miniML event window size, by default 3600.
    direction : str, optional
        Event direction for detection, by default 'negative'.
    bla_epoch_idx : int, optional
        Epoch index of the BLA stimulation onset, by default 5.
    ca3_epoch_idx : int, optional
        Epoch index of the CA3 stimulation onset, by default 6.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Per-sweep spontaneous event dictionaries.

    Raises
    ------
    ValueError
        If epoch indices are invalid for the ABF protocol.

    """

    if all_events is None:
        if model is None:
            raise ValueError("Provide either `model` or `all_events`.")

        all_events = detect_all_events(
            ephy_instance=ephy_instance,
            file_path=file_path,
            model=model,
            channel=channel,
            scaling=scaling,
            unit=unit,
            win_size=win_size,
            direction=direction)

    abf = pyabf.ABF(file_path)
    epoch_starts = abf.sweepEpochs.p1s

    max_epoch_idx = max(bla_epoch_idx, ca3_epoch_idx)
    if len(epoch_starts) <= max_epoch_idx:
        raise ValueError(
            "Epoch index out of range. "
            f"Requested BLA={bla_epoch_idx}, CA3={ca3_epoch_idx}, "
            f"available epochs={len(epoch_starts)}")

    evoked_window_s = _get_evoked_window_duration_s(ephy_instance)

    bla_start_s = epoch_starts[bla_epoch_idx] / abf.sampleRate
    ca3_start_s = epoch_starts[ca3_epoch_idx] / abf.sampleRate
    sweep_end_s = abf.sweepPointCount / abf.sampleRate

    bla_end_s = bla_start_s + evoked_window_s
    ca3_end_s = ca3_start_s + evoked_window_s

    spontaneous_windows = [
        (bla_end_s, ca3_start_s),
        (ca3_end_s, sweep_end_s)
    ]

    spontaneous_events: list[dict[str, np.ndarray]] = []
    for sweep_events in all_events:
        event_indices = _find_events_in_windows(
            peak_x=sweep_events["peak_x"],
            windows_s=spontaneous_windows)
        sweep_spontaneous = _subset_sweep_events(
            sweep_events=sweep_events,
            event_indices=event_indices)
        spontaneous_events.append(sweep_spontaneous)

    return spontaneous_events
