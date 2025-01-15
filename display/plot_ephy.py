""" Created on Wed Jan 24 09:37:42 2024
    @author: dcupolillo """

import matplotlib.pyplot as plt
import pyabf
import numpy as np
import warnings


def plot_current_pulses(
        trace: pyabf.ABF,
        command: bool = None,
        trace_color: str = None,
        alpha: float = None,
        command_color: str = None,
        ax: plt.Axes = None
) -> None:

    rows = 1 + int(command)

    if not ax:
        fig_size = (6, 5 * rows)
        fig, ax = plt.subplots(rows, 1, figsize=fig_size, sharex=True)

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for sweep_n in trace.sweepList:
        trace.setSweep(sweep_n)
        ax[0].plot(
            trace.sweepX,
            trace.sweepY,
            color=trace_color,
            alpha=alpha)
        ax[0].set_ylabel(trace.sweepLabelY)

    if command:
        for sweep_n in trace.sweepList:
            trace.setSweep(sweep_n)
            ax[1].plot(
                trace.sweepX,
                trace.sweepC,
                color=command_color)
            ax[1].set_ylabel(trace.sweepLabelC)

    ax[-1].set_xlabel(trace.sweepLabelX)
    plt.tight_layout()


def plot_all_sweeps_electrophysiology(
        trace: pyabf.ABF,
        ax: plt.Axes = None,
        trace_color: str = None,
        channels: int or list or np.ndarray = None,
        alpha: float = None,
        mean: bool = None,
        mean_color: str = None,
        command: bool = None,
        command_color: str = None,
        dacs: int or list or np.ndarray = None,
        channels_palette: str = None,
        dacs_palette: str = None
) -> None:

    if isinstance(channels, int):
        channels = [channels]

    available_channels = range(trace.channelCount)
    if len(channels) > len(available_channels):
        warnings.warn(f'Only {len(available_channels)} channels available')
    channels = [ch for ch in channels if ch in available_channels]
    if not channels:
        raise ValueError("No valid channels provided")

    rows = len(channels) + int(command) + int(dacs is not None)

    if not ax:
        fig_size = (8, 3 * rows)
        fig, ax = plt.subplots(rows, 1, figsize=fig_size, sharex=True)
        ax[-1].set_xlabel(trace.sweepLabelX)

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    channel_color = (plt.cm.get_cmap(channels_palette)(np.linspace(
        0, 1, len(channels))) if len(channels) > 1 else [trace_color])

    for i, channel in enumerate(channels):
        for sweep_n in trace.sweepList:
            trace.setSweep(sweep_n, channel=channel)
            ax[i].plot(
                trace.sweepX,
                trace.sweepY,
                color=channel_color[i],
                alpha=alpha)
        # ax[i].set_ylabel(f'Channel {channel} - {trace.sweepLabelY}')

    if mean:
        traces_array = [None] * len(trace.sweepList)
        for sweep_n in trace.sweepList:
            trace.setSweep(sweep_n)
            traces_array[sweep_n] = trace.sweepY
        mean_trace = np.mean(traces_array, axis=0)
        ax[0].plot(trace.sweepX, mean_trace, color=mean_color)

    if command:
        command_index = 1 if len(channels) == 1 else len(channels)
        trace.setSweep(0, channel=0)
        ax[command_index].set_ylabel(trace.sweepLabelC)
        ax[command_index].plot(
            trace.sweepX,
            trace.sweepC,
            color=command_color)

    if dacs:
        dac_index = len(channels) + int(command)
        dac_colors = (plt.cm.get_cmap(dacs_palette)(np.linspace(
            0, 1, len(dacs))) if len(dacs) > 1 else ['blue'])

        if len(dacs) > 1:
            dac_colors = plt.cm.viridis(np.linspace(0, 1, len(dacs)))
        else:
            dac_colors = ['blue']

        for i, dac in enumerate(dacs):
            trace.setSweep(0, channel=0)
            ax[dac_index].plot(
                trace.sweepX,
                trace.sweepD(dac),
                color=dac_colors[i],
                label=f'DAC {dac}')
            ax[dac_index].legend()

        ax[dac_index].set_ylabel('Digital Output')
        ax[dac_index].set_yticks([0, 1])
        ax[dac_index].set_yticklabels(["OFF", "ON"])
        ax[dac_index].axes.set_ylim(-.5, 1.5)

    plt.tight_layout()


def plot_sweep_electrophysiology(
        trace: pyabf.ABF,
        sweep_index: int,
        trace_color: str,
        channels: int or list or np.ndarray,
        command: bool,
        command_color: str,
        dacs: int or list or np.ndarray,
        channels_palette: str,
        dacs_palette: str
) -> None:

    if isinstance(channels, int):
        channels = [channels]
    if isinstance(dacs, int):
        dacs = [dacs]

    available_channels = range(trace.channelCount)
    if len(channels) > len(available_channels):
        warnings.warn(f'Only {len(available_channels)} channels available')
    channels = [ch for ch in channels if ch in available_channels]
    if not channels:
        raise ValueError("No valid channels provided")

    rows = len(channels) + int(command) + int(dacs is not None)

    fig_size = (8, 3 * rows)

    fig, ax = plt.subplots(rows, 1, figsize=fig_size, sharex=True)

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    channel_color = (plt.cm.get_cmap(channels_palette)(np.linspace(
        0, 1, len(channels))) if len(channels) > 1 else [trace_color])

    for i, channel in enumerate(channels):
        trace.setSweep(sweep_index, channel=channel)
        ax[i].set_ylabel(f'Channel {channel} - {trace.sweepLabelY}')
        ax[i].plot(trace.sweepX, trace.sweepY, color=channel_color[i])

    if command:
        command_index = 1 if len(channels) == 1 else len(channels)
        trace.setSweep(sweep_index, channel=0)
        ax[command_index].set_ylabel(trace.sweepLabelC)
        ax[command_index].plot(
            trace.sweepX,
            trace.sweepC,
            color=command_color)

    if dacs:
        dac_index = len(channels) + int(command)
        dac_colors = (plt.cm.get_cmap(dacs_palette)(np.linspace(
            0, 1, len(dacs))) if len(dacs) > 1 else ['blue'])

        if len(dacs) > 1:
            dac_colors = plt.cm.viridis(np.linspace(0, 1, len(dacs)))
        else:
            dac_colors = ['blue']

        for i, dac in enumerate(dacs):
            trace.setSweep(sweep_index, channel=0)
            ax[dac_index].plot(
                trace.sweepX,
                trace.sweepD(dac),
                color=dac_colors[i],
                label=f'DAC {dac}')
            ax[dac_index].legend()

        ax[dac_index].set_ylabel('Digital Output')
        ax[dac_index].set_yticks([0, 1])
        ax[dac_index].set_yticklabels(["OFF", "ON"])
        ax[dac_index].axes.set_ylim(-.5, 1.5)

    ax[-1].set_xlabel(trace.sweepLabelX)
    plt.tight_layout()
