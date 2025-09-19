""" Created on Tue Aug 8 13:23:29 2025
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt


def plot_traces(
        roi_segmenter,
        input_type: str = 'BLA',
        fontsize: int = 10,
        increment: float = 2.0,
        spines_cmap: str = 'gist_rainbow',
        sweep_color: str = 'gray',
        sweep_alpha: float = 0.5,
        sweep_linewidth: float = 1.0,
        trace_linewidth: float = 1.5,
        scalebar_y_unit: float = 0.5,
        ax: plt.Axes = None,
        **kwargs
) -> plt.Axes:
    """
    Plot dF/F calcium traces for spines in an ROI.

    Parameters
    ----------
    roi_segmenter : RoiSegmenter
        ROI segmenter instance containing spine data and traces.
    input_type : str, optional
        Data source ('BLA' or 'CA3'). Default is 'BLA'.
    fontsize : int, optional
        Font size for labels and text. Default is 10.
    increment : float, optional
        Vertical spacing between traces. Default is 2.0.
    spines_cmap : str, optional
        Colormap for spine traces. Default is 'gist_rainbow'.
    sweep_color : str, optional
        Color for individual sweep traces. Default is 'gray'.
    sweep_alpha : float, optional
        Transparency for sweep traces. Default is 0.5.
    sweep_linewidth : float, optional
        Line width for sweep traces. Default is 1.0.
    trace_linewidth : float, optional
        Line width for mean traces. Default is 1.5.
    scalebar_y_unit : float, optional
        Y-axis scale bar unit. Default is 0.5.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    **kwargs
        Additional arguments for plot customization.

    Returns
    -------
    plt.Axes
        The matplotlib axes with plotted traces.
    
    Raises
    ------
    ValueError
        If no spines are available or invalid input_type.
    """
    if roi_segmenter.n_spines == 0:
        raise ValueError("No spines available to plot.")
        
    if input_type not in ['BLA', 'CA3']:
        raise ValueError("input_type must be 'BLA' or 'CA3'")

    # Get data based on input type
    if input_type == 'BLA':
        dff_data = roi_segmenter.dFF_BLA
        ts_data = roi_segmenter.ts_BLA
    else:
        dff_data = roi_segmenter.dFF_CA3
        ts_data = roi_segmenter.ts_CA3

    if len(dff_data) == 0:
        raise ValueError(f"No {input_type} data available for plotting.")

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    
    # Get colormap for spines
    cmap = plt.get_cmap(spines_cmap, roi_segmenter.n_spines)
    
    # Plot traces for each spine
    for spine_idx in range(roi_segmenter.n_spines):
        color = cmap(spine_idx)
        y_offset = spine_idx * increment
        
        spine_dff = dff_data[spine_idx]
        spine_ts = ts_data[spine_idx]
        
        # Plot individual sweeps
        if len(spine_dff.shape) > 1:  # Multiple sweeps
            for sweep in range(spine_dff.shape[0]):
                ax.plot(
                    spine_ts, 
                    spine_dff[sweep] + y_offset,
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth
                )
            
            # Plot mean trace
            mean_trace = np.mean(spine_dff, axis=0)
            ax.plot(
                spine_ts,
                mean_trace + y_offset,
                color=color,
                linewidth=trace_linewidth,
                label=f'Spine {spine_idx + 1}'
            )
        else:  # Single trace
            ax.plot(
                spine_ts,
                spine_dff + y_offset,
                color=color,
                linewidth=trace_linewidth,
                label=f'Spine {spine_idx + 1}'
            )

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_ylabel('dF/F + offset', fontsize=fontsize)
    ax.tick_params(both='major', labelsize=fontsize)
    
    # Add scale bar
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8)
    ax.text(0, -increment/2, f'{scalebar_y_unit} dF/F', 
            fontsize=fontsize, ha='left', va='top')
    
    return ax


def plot_zscores(
        roi_segmenter,
        input_type: str = 'CA3',
        zscore_cmap: str = 'viridis',
        fontsize: int = 10,
        ax: plt.Axes = None,
        colorbar_kwargs: dict = None,
        **kwargs
) -> plt.Axes:
    """
    Plot z-scores as a heatmap for spines in an ROI.

    Parameters
    ----------
    roi_segmenter : RoiSegmenter
        ROI segmenter instance containing spine data and z-scores.
    input_type : str, optional
        Data source ('BLA' or 'CA3'). Default is 'CA3'.
    zscore_cmap : str, optional
        Colormap for z-score visualization. Default is 'viridis'.
    fontsize : int, optional
        Font size for labels. Default is 10.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    colorbar_kwargs : dict, optional
        Additional arguments for colorbar customization.
    **kwargs
        Additional arguments passed to imshow.

    Returns
    -------
    plt.Axes
        The matplotlib axes with plotted z-scores.
        
    Raises
    ------
    ValueError
        If no spines are available or invalid input_type.
    """
    if roi_segmenter.n_spines == 0:
        raise ValueError("No spines available to plot.")
        
    if input_type not in ['BLA', 'CA3']:
        raise ValueError("input_type must be 'BLA' or 'CA3'")

    # Get data based on input type
    if input_type == 'BLA':
        zscore_data = roi_segmenter.zscore_BLA
        ts_data = roi_segmenter.ts_BLA
    else:
        zscore_data = roi_segmenter.zscore_CA3
        ts_data = roi_segmenter.ts_CA3

    if len(zscore_data) == 0:
        raise ValueError(f"No {input_type} z-score data available for plotting.")

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")

    # Prepare colorbar kwargs
    if colorbar_kwargs is None:
        colorbar_kwargs = {}
    
    cbar_defaults = {
        'label': 'Z-score',
        'shrink': 0.8
    }
    cbar_params = {**cbar_defaults, **colorbar_kwargs}

    # Convert to array for plotting
    zscore_array = np.array(zscore_data)
    
    # Handle different data shapes
    if len(zscore_array.shape) == 3:  # Multiple sweeps
        zscore_array = np.mean(zscore_array, axis=1)  # Average across sweeps
    
    # Create heatmap
    im = ax.imshow(
        zscore_array, 
        cmap=zscore_cmap,
        aspect='auto',
        **kwargs
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, **cbar_params)
    
    # Set labels and formatting
    ax.set_xlabel('Time (frames)', fontsize=fontsize)
    ax.set_ylabel('Spine #', fontsize=fontsize)
    ax.tick_params(both='major', labelsize=fontsize)
    
    # Set spine labels
    spine_labels = [f'{i+1}' for i in range(roi_segmenter.n_spines)]
    ax.set_yticks(range(roi_segmenter.n_spines))
    ax.set_yticklabels(spine_labels)
    
    return ax


def plot_single(
        spine,
        input_type: str = 'BLA',
        fontsize: int = 10,
        increment: float = 1.0,
        sweep_color: str = 'gray',
        sweep_alpha: float = 0.5,
        mean_color: str = 'green',
        mean_alpha: float = 1.0,
        sweep_linewidth: float = 1.0,
        mean_linewidth: float = 1.5,
        scalebar_y_unit: float = 0.5,
        ax: plt.Axes = None,
        **kwargs
) -> plt.Axes:
    """
    Plot dF/F traces for a single spine.

    Parameters
    ----------
    spine : Spine
        Spine instance containing trace data.
    input_type : str, optional
        Data source ('BLA' or 'CA3'). Default is 'BLA'.
    fontsize : int, optional
        Font size for labels. Default is 10.
    increment : float, optional
        Vertical spacing between sweeps. Default is 1.0.
    sweep_color : str, optional
        Color for individual sweep traces. Default is 'gray'.
    sweep_alpha : float, optional
        Transparency for sweep traces. Default is 0.5.
    mean_color : str, optional
        Color for mean trace. Default is 'green'.
    mean_alpha : float, optional
        Transparency for mean trace. Default is 1.0.
    sweep_linewidth : float, optional
        Line width for sweep traces. Default is 1.0.
    mean_linewidth : float, optional
        Line width for mean trace. Default is 1.5.
    scalebar_y_unit : float, optional
        Y-axis scale bar unit. Default is 0.5.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    **kwargs
        Additional arguments for plot customization.

    Returns
    -------
    plt.Axes
        The matplotlib axes with plotted traces.
        
    Raises
    ------
    ValueError
        If invalid input_type or no data available.
    """
    if input_type not in ['BLA', 'CA3']:
        raise ValueError("input_type must be 'BLA' or 'CA3'")

    # Get data based on input type
    if input_type == 'BLA':
        dff_data = spine.dFF_BLA
        ts_data = spine.ts_BLA
    else:
        dff_data = spine.dFF_CA3
        ts_data = spine.ts_CA3

    if len(dff_data) == 0:
        raise ValueError(f"No {input_type} data available for spine.")

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")

    # Plot individual sweeps if multiple available
    if len(dff_data.shape) > 1:  # Multiple sweeps
        for sweep_idx in range(dff_data.shape[0]):
            y_offset = sweep_idx * increment
            ax.plot(
                ts_data,
                dff_data[sweep_idx] + y_offset,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                label=f'Sweep {sweep_idx + 1}' if sweep_idx < 5 else None
            )
        
        # Plot mean trace
        mean_trace = np.mean(dff_data, axis=0)
        mean_offset = dff_data.shape[0] * increment
        ax.plot(
            ts_data,
            mean_trace + mean_offset,
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            label='Mean'
        )
    else:  # Single trace
        ax.plot(
            ts_data,
            dff_data,
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            label='dF/F'
        )

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_ylabel('dF/F + offset', fontsize=fontsize)
    ax.tick_params(both='major', labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    
    # Add scale bar
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8)
    ax.text(0, -increment/2, f'{scalebar_y_unit} dF/F', 
            fontsize=fontsize, ha='left', va='top')
    
    return ax
