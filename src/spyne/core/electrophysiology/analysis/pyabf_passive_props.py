""" Created on Tue Sep 23 13:12:00 2025
    @author: dcupolillo """

from __future__ import annotations
import numpy as np
import pyabf
from scipy.optimize import curve_fit
from ipfx.feature_extractor import SpikeFeatureExtractor


def ohm_law(
        voltage: float | None = None,
        resistance: float | None = None,
        current: float | None = None
) -> float:
    """
    Calculate voltage, current, or resistance using Ohm's Law.
    V = I * R
    I = V / R
    R = V / I
    
    Parameters
    ----------
    voltage : float, optional
        The voltage in volts (V). Default is None.
    resistance : float, optional
        The resistance in ohms (Ω). Default is None.
    current : float, optional
        The current in amperes (A). Default is None.
    Returns
    -------
    float
        The calculated value (voltage, current, or resistance).
        Since 1 ohm is defined as 1 volt divided by 1 ampere (1 Ω = 1 V/A):
            - Resistance is returned in ohms (Ω).
            - Voltage is returned in volts (V).
            - Current is returned in amperes (A).
    Raises
    ------
    ValueError
        If less than two parameters are provided.
    """
    if voltage is not None and resistance is not None:
        return voltage / resistance
    elif voltage is not None and current is not None:
        return voltage / current
    elif current is not None and resistance is not None:
        return current * resistance
    else:
        raise ValueError(
            "At least one of voltage, current, or resistance must be provided."
            )


def exp_decay(
        t: float,
        A: float,
        tau: float,
        C: float
) -> float:
    
    """Single exponential decay.
    
    Parameters
    ----------
    t : float
        Time variable.
    A : float
        Amplitude of the exponential decay.
    tau : float
        Time constant of the decay.
    C : float
        Offset value.
    Returns
    -------
    float
        The value of the exponential decay at time t."""
    
    return A * np.exp(-t / tau) + C


def fit_to_exp_decay(
        t: np.ndarray,
        y: np.ndarray,
        A0: float,
        tau0: float,
        C0: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit data to a single exponential decay model.

    Parameters
    ----------
    t : np.ndarray
        The time data array.
    y : np.ndarray
        The data array to fit.
    A0 : float
        Initial guess for amplitude.
    tau0 : float
        Initial guess for time constant.
    C0 : float
        Initial guess for offset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the optimal parameters and the covariance of the parameters.
    """
    popt, pcov = curve_fit(exp_decay, t, y, p0=[A0, tau0, C0])
    
    return popt, pcov


def calculate_baseline(
        data: np.ndarray,
        window: tuple[int, int] = (0, 1000)
) -> float: 
    """
    Calculate the baseline from the provided data.

    Parameters
    ----------
    data : np.ndarray
        The input data array containing voltage measurements.
    window : tuple[int, int], optional
        The time window (start, end) to use for baseline calculation.
        Default is (0, 1000).

    Returns
    -------
    float
        The calculated baseline value.
    """

    return data[window[0]:window[1]].mean()


def analyze_test_pulse(
        data: np.ndarray,
        pulse_window: tuple[int, int],
        pulse_level: float,
        min_percentage: float = 10.0,
        max_percentage: float = 90.0,
        sample_rate: float = 10000.0,
        A0: float = None,
        tau0: float = None,
        C0: float = None,
    ) -> dict:
    """
    Analyze the test pulse in the provided data.

    Parameters
    ----------
    data : np.ndarray
        The input data array containing voltage measurements.
    pulse_window : tuple[int, int]
        The time window (start, end) to analyze the test pulse.
    pulse_level : float
        The level of the test pulse in picoamperes (pA).
    sample_rate : float, optional
        The sample rate of the data in Hz. Default is 10000.0.
    A0 : float, optional
        Initial guess for amplitude of exponential decay. Default is None.
    tau0 : float, optional
        Initial guess for time constant of exponential decay. Default is None.
    C0 : float, optional
        Initial guess for offset of exponential decay. Default is None.

    Returns
    -------
    dict
        A dictionary containing analysis results such as peak amplitude,
        rise time, and decay time.
    """
    pulse_level *= 1e-3  # Convert mV to V

    # Assumes baseline from start of sweep to pulse onset
    baseline = calculate_baseline(data, (0, pulse_window[0]))
    pulse_data = data[pulse_window[0]:pulse_window[1]] - baseline

    # Upsample pulse_data to 200 kHz for accurate threshold detection
    current_sampling_rate = sample_rate
    target_sampling_rate = 200_000  # Hz
    current_sampling = 1 / current_sampling_rate
    target_sampling = 1 / target_sampling_rate
    time_ax_original = np.arange(0, pulse_data.shape[0]) * current_sampling
    resampled_time_ax = np.arange(0, time_ax_original[-1] + target_sampling, target_sampling)
    pulse_data_resampled = np.interp(resampled_time_ax, time_ax_original, pulse_data)

    amplitude = pulse_data_resampled.max()
    amplitude_index = np.argmax(pulse_data_resampled)

    # Slice out the rise segment to isolate decay
    pulse_data_decay = pulse_data_resampled[amplitude_index:]
    resampled_time_ax_decay = (
        resampled_time_ax[amplitude_index:] -
        resampled_time_ax[amplitude_index])
    
    pulse_data_rise = pulse_data_resampled[:amplitude_index+1]
    resampled_time_ax_rise = (
        resampled_time_ax[:amplitude_index+1] -
        resampled_time_ax[amplitude_index])

    # Calculate amplitude and thresholds using baseline and amplitude
    min_percentage = 10
    max_percentage = 90
    min_level = amplitude * min_percentage / 100
    max_level = amplitude * max_percentage / 100

    # Find indices for decay segment
    decay_start = np.argmax(pulse_data_decay >= max_level)
    decay_end = np.argmax(pulse_data_decay <= min_level)
    decay_segment = pulse_data_decay[decay_start:decay_end]
    decay_time_axis = resampled_time_ax_decay[decay_start:decay_end]
    t = np.arange(len(decay_segment)) * target_sampling  # time axis for resampled data

    # Guard clause for empty decay segment
    if decay_segment.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Initial parameter guesses
    if A0 is None:
        A0 = decay_segment.max() - decay_segment.min()
    if tau0 is None:
        tau0 = (decay_end - decay_start) * target_sampling / 2
    if C0 is None:
        C0 = decay_segment.min()

    popt, pcov = fit_to_exp_decay(t, decay_segment, A0, tau0, C0)
    A_fit, tau_fit, C_fit = popt

    tau_sec = (1 / tau_fit) * sample_rate

    # Use resampled data and time axis for rise/decay time calculations
    rise_start_idx = np.argmax(pulse_data_rise >= min_level)
    rise_end_idx = np.argmax(pulse_data_rise >= max_level)
    rise_time = resampled_time_ax_rise[rise_end_idx] - resampled_time_ax_rise[rise_start_idx]

    decay_start_idx = np.argmax(pulse_data_decay >= max_level)
    decay_end_idx = np.argmax(pulse_data_decay <= min_level)
    decay_time = (
        resampled_time_ax_decay[decay_end_idx] -
        resampled_time_ax_decay[decay_start_idx])

    # Get access resistance
    amplitude_in_ampere = amplitude * 1e-12
    Ra = ohm_law(voltage=pulse_level, current=amplitude_in_ampere)

    # Get steady state (the last 20% of the pulse)  
    Iss = pulse_data[int(0.8 * len(pulse_data)) :].mean() - baseline

    # Get membrane resistance
    Iss_in_ampere = Iss * 1e-12 # Convert pA to A
    Rm = ohm_law(voltage=pulse_level, current=Iss_in_ampere)

    return Ra, Rm, Iss, rise_time, decay_time, tau_sec


def passive_properties(
        sweep_y: np.ndarray,
        test_pulse_start: int,
        test_pulse_end: int,
        test_pulse_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to calculate passive properties across multiple sweeps.
    
    Parameters
    ----------
    sweep_y : np.ndarray
        3D array of shape (n_recordings, n_sweeps, n_timepoints)
        containing the voltage traces.
    test_pulse_start : int
        The start index of the test pulse.
    test_pulse_end : int
        The end index of the test pulse.
    test_pulse_level : float
        The level of the test pulse in millivolts (mV).
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuples of arrays containing Ih, Ra, Rm, and Iss values.
    """

    n_recordings, n_sweeps, _ = sweep_y.shape
    Ih = np.empty(n_recordings, dtype=object)
    Ra = np.empty(n_recordings, dtype=object)
    Rm = np.empty(n_recordings, dtype=object)
    Iss = np.empty(n_recordings, dtype=object)
    rise_time = np.empty(n_recordings, dtype=object)
    decay_time = np.empty(n_recordings, dtype=object)
    tau_sec = np.empty(n_recordings, dtype=object)

    for z_plane in range(n_recordings):
        
        Ih[z_plane] = np.empty(n_sweeps, dtype=float)
        Ra[z_plane] = np.empty(n_sweeps, dtype=float)
        Rm[z_plane] = np.empty(n_sweeps, dtype=float)
        Iss[z_plane] = np.empty(n_sweeps, dtype=float)
        rise_time[z_plane] = np.empty(n_sweeps, dtype=float)
        decay_time[z_plane] = np.empty(n_sweeps, dtype=float)
        tau_sec[z_plane] = np.empty(n_sweeps, dtype=float)

        for n_sweep in range(n_sweeps):
            Ih[z_plane][n_sweep] = calculate_baseline(
                sweep_y[z_plane][n_sweep],
                (0, test_pulse_start))
            Ra[z_plane][n_sweep], Rm[z_plane][n_sweep], Iss[z_plane][n_sweep], \
                rise_time[z_plane][n_sweep], decay_time[z_plane][n_sweep], \
                    tau_sec[z_plane][n_sweep] = analyze_test_pulse(
                sweep_y[z_plane][n_sweep],
                (test_pulse_start, test_pulse_end),
                test_pulse_level,
                (0, test_pulse_start))
            
    # Stack arrays to ensure consistent shape (convert from object arrays to 2D arrays)
    Ih = np.vstack(Ih)
    Ra = np.vstack(Ra)
    Rm = np.vstack(Rm)
    Iss = np.vstack(Iss)
    rise_time = np.vstack(rise_time)
    decay_time = np.vstack(decay_time)
    tau_sec = np.vstack(tau_sec)
            
    return Ih, Ra, Rm, Iss, rise_time, decay_time, tau_sec

    
def analyze_I_steps(
        filen_name: str or Path,
) -> tuple:
    """
    Analyze I-V relationship and firing frequency from an I-step protocol ABF file.

    Parameters
    ----------
    filen_name : str or Path
        Path to the ABF file.
    Returns
    -------
    tuple
        A tuple containing:
        - Isteps: np.ndarray of current steps in pA.
        - IF: np.ndarray of firing frequencies in Hz.
        - IV: np.ndarray of steady-state voltage changes in mV.
        - spike_dfs: np.ndarray of DataFrames containing spike features for each sweep.
    """

    abf = pyabf.ABF(filen_name)

    ext = SpikeFeatureExtractor(filter=1)

    if not abf.sweepUnitsY == 'mV':
        return None, None, None, None

    epochs = abf.sweepEpochs.p1s
    pulse_window = (epochs[2], epochs[3])  # Assuming the 3rd epoch is the pulse
    pulse_duration = (pulse_window[1] - pulse_window[0]) / abf.sampleRate

    Isteps = np.empty(len(abf.sweepList), dtype=float)
    IF = np.zeros(len(abf.sweepList), dtype=int)
    IV = np.full(len(abf.sweepList), np.nan, dtype=float)
    spike_dfs = np.full(len(abf.sweepList), None, dtype=object)

    for sweep_n in abf.sweepList:
        abf.setSweep(sweep_n)

        Isteps[sweep_n] = np.mean(abf.sweepC[pulse_window[0]:pulse_window[1]])
        baseline = np.mean(abf.sweepY[:pulse_window[0]])

        spikes = ext.process(abf.sweepX, abf.sweepY, abf.sweepC)
        # Store the DataFrame (or None) for this sweep
        spike_dfs[sweep_n] = spikes

        if spikes is not None and hasattr(spikes, 'shape') and spikes.shape[0] > 0:
            n_spikes = spikes.shape[0]  # (current step, number of spikes)
            spike_frequency = n_spikes / pulse_duration
            IF[sweep_n] = spike_frequency
        else:
            # Last 25% of the pulse
            segment = abf.sweepY[int(0.75 * pulse_window[0]):pulse_window[1]]
            steady_state = np.mean(segment)
            dV = steady_state - baseline
            IV[sweep_n] = dV

    return Isteps, IF, IV, spike_dfs


def unpack_spike_dfs(
        spike_dfs: np.ndarray,
) -> tuple:
    """
    Unpack spike feature DataFrames into separate arrays for each feature.

    Parameters
    ----------
    spike_dfs : np.ndarray
        Array of DataFrames containing spike features for each sweep.
    Returns
    -------
    tuple
        A tuple containing arrays for each spike feature:
        - peaks_v: Peak voltages of spikes.
        - peaks_t: Times of peak voltages.
        - threshold_t: Times of spike thresholds.
        - threshold_v: Voltages at spike thresholds.
        - upstroke_t: Times of spike upstrokes.
        - upstroke_v: Voltages at spike upstrokes.
        - downstroke_t: Times of spike downstrokes.
        - downstroke_v: Voltages at spike downstrokes.
        - width: Widths of spikes.
    """
    # Guard clause for None input
    if spike_dfs is None:
        empty = []
        return (empty, empty, empty, empty, empty, empty, empty, empty, empty, empty)

    peaks_v = []
    peaks_t = []
    threshold_t = []
    threshold_v = []
    upstroke_t = []
    upstroke_v = []
    downstroke_t = []
    downstroke_v = []
    width = []
    amplitude = []

    for df in spike_dfs:
        if df is not None and not df.empty:
            # Extract all columns as before
            for col, out_list in [
                ('peak_v', peaks_v),
                ('peak_t', peaks_t),
                ('threshold_t', threshold_t),
                ('threshold_v', threshold_v),
                ('upstroke_t', upstroke_t),
                ('upstroke_v', upstroke_v),
                ('downstroke_t', downstroke_t),
                ('downstroke_v', downstroke_v),
                ('width', width),
            ]:
                if col in df.columns:
                    out_list.append(df[col].to_numpy())
                else:
                    out_list.append(np.full(df.shape[0], np.nan))
            # Amplitude: peak_v - threshold_v
            if 'peak_v' in df.columns and 'threshold_v' in df.columns:
                amp = df['peak_v'].to_numpy() - df['threshold_v'].to_numpy()
                amplitude.append(amp)
            else:
                amplitude.append(np.full(df.shape[0], np.nan))
        else:
            peaks_v.append(np.array([]))
            amplitude.append(np.array([]))

    return (
        peaks_v, peaks_t,
        threshold_t, threshold_v,
        upstroke_t, upstroke_v,
        downstroke_t, downstroke_v,
        width,
        amplitude)