import pyabf
import numpy as np
import tensorflow as tf
import miniML
from miniML.core.miniML_plot_functions import miniML_plots


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


def BLA_EPSCs(
        ephy_instance,
        file_path,
        model: tf.keras.models.Model,
        channel: int = 0,
        scaling: int = 1,
        unit: str = 'pA',
        win_size: int = 3600,
        direction: str = 'negative'
) -> tuple:
    """
    Analyze EPSCs in BLA neurons using miniML.
    Parameters
    ----------
    ephy_instance : Ephydataset
        The Ephydataset instance containing the data.
    file_path : str
        Path to the ABF file.
    channel : int, optional
        The channel index to analyze. Default is 0.
    scaling : int, optional
        Scaling factor for the data. Default is 1.
    unit : str, optional
        Unit of the data. Default is 'pA'.
    win_size : int, optional
        Window size for event detection. Default is 800.
    direction : str, optional
        Direction of events to detect ('negative' or 'positive'). Default is 'negative'.
    Returns
    -------
    tuple
        A tuple containing arrays of detected events and their properties.
    """

    abf = pyabf.ABF(file_path)
    n_sweeps = len(abf.sweepList)

    dataset = ephy_instance._dataset

    n_frames = dataset[0].n_frames
    frame_rate = dataset[0].frame_rate
    duration = n_frames * (1 / frame_rate)
    sampling_points = int(duration * abf.sampleRate)

    add_points = int(win_size / 3)
    event_len = add_points + win_size + add_points

    (events, events_x,
     amplitudes, peak_values, peak_x,
     bsl_start_x, bsl_end_x,
     scores, tau, charges,
     risetimes, rise_min_x, rise_max_x,
     slopes, decaytimes, halfwidths) = init_containers(n_sweeps, event_len)

    for sweep_n in abf.sweepList:
        abf.setSweep(sweep_n)

        # load from ABF file
        trace = miniML.MiniTrace.from_axon_file(
            filename=file_path,
            channel=channel,
            sweep=sweep_n,
            scaling=scaling,
            unit=unit)

        # BLA trimming
        t1 = abf.sweepEpochs.p1s[5]
        t2 = int(t1 + sampling_points)

        trace.data = abf.sweepY[t1:t2]

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
        
        # print(detector.event_len)
        
        if detector.events is not None and detector.events.shape[0] > 0:
            
            idx = 0
            ev = np.asarray(detector.events[idx], dtype=float)
            
            # assume ev.size == event_len; if not, pad/crop as needed
            events[sweep_n, :] = ev
            events_x[sweep_n, :] = np.arange(event_len) * detector.trace.sampling

            amplitudes[sweep_n] = detector.event_peak_values[idx] - detector.event_bsls[idx]
            peak_values[sweep_n] = detector.event_peak_values[idx]
            peak_x[sweep_n] = detector.event_peak_times[idx]  # convert to time if needed

            bsl_start_x[sweep_n] = detector.bsl_starts[idx] * detector.trace.sampling
            bsl_end_x[sweep_n] = detector.bsl_ends[idx] * detector.trace.sampling

            risetimes[sweep_n] = detector.risetimes[idx]
            rise_min_x[sweep_n] = detector.min_positions_rise[idx] * detector.trace.sampling
            rise_max_x[sweep_n] = detector.max_positions_rise[idx] * detector.trace.sampling
            scores[sweep_n] = (detector.event_scores[idx]
                               if detector.event_scores is not None else np.nan)
            
            tau[sweep_n] = (detector.avg_decay_fit[1]
                            if detector.avg_decay_fit is not None else np.nan)
            
            charges[sweep_n] = (detector.charges[idx]
                                if detector.charges is not None else np.nan)
            slopes[sweep_n] = (detector.slopes[idx]
                               if detector.slopes is not None else np.nan)
            decaytimes[sweep_n] = (detector.decaytimes[idx]
                                   if detector.decaytimes is not None else np.nan)
            halfwidths[sweep_n] = (detector.halfwidths[idx]
                                   if detector.halfwidths is not None else np.nan)
    
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
