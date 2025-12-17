""" Created on Wed Sep  6 11:31:57 2023
    @author: dcupolillo 
    
    Input/Output functions for imaging data.
    """

from pathlib import Path
import cv2
import numpy as np
import flammkuchen as fl
import time


def animate_frames(
    frames: np.ndarray,
    frame_rate: float,
    active_channels: list,
    timestamps: bool,
    norm: list or tuple or np.ndarray = None
) -> None:
    """
    Display a sequence of frames as an animated loop.

    This function visualizes imaging frames using OpenCV,
    allowing for side-by-side display of multiple active channels
    and optional timestamp annotations.

    Parameters
    ----------
    frames : np.ndarray
        A sequence of frames to display, either as single-channel or
        multi-channel. For multi-channel input, frames should have
        shape (n_frames, n_channels, height, width).
    frame_rate : float
        The display rate of the frames, in frames per second.
    active_channels : list, optional
        A list of channel indices to display. If specified, frames from these
        channels are displayed side-by-side.
    timestamps : bool, optional
        If True, elapsed time is displayed on the frames during the animation.
    norm : list | tuple | np.ndarray, optional
        Normalization range for frame intensity values as [min, max]. If None,
        the default range [0, 255] is used.

    Returns
    -------
    None
        The function displays frames in an OpenCV window. Press 'Q' to exit
        the animation loop.

    Notes
    -----
    - The function automatically normalizes frames to the 8-bit range
        for displaying.
    - The display window is upscaled for better visibility.
    - Timestamps are added to frames based on the provided frame rate.

    """

    n_frames = len(frames)
    frame_period = 1 / frame_rate
    scan_duration = frame_period * n_frames

    if timestamps:
        timestamps = np.linspace(0, scan_duration, n_frames)

    vmin, vmax = norm if norm else (0, 255)

    if not active_channels:
        frame_height, frame_width = frames[0].shape
        
        # Pre-allocate normalized frames array for single channel
        normalized_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            normalized_frames[i] = cv2.normalize(
                frame, None, vmin, vmax, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frames = normalized_frames
    
    else:
        # Get dimensions for multi-channel combination
        _, _, frame_height, single_width = frames.shape
        frame_width = single_width * len(active_channels)
        
        # Pre-allocate combined frames array
        combined_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.uint8)

        for i in range(n_frames):
            for n, channel in enumerate(active_channels):
                # Extract and normalize the channel frame
                channel_frame = frames[i, n, :, :]
                frame_normalized = cv2.normalize(
                    channel_frame, None, vmin, vmax, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Place in the correct position in combined frame
                start_col = n * single_width
                end_col = (n + 1) * single_width
                combined_frames[i, :, start_col:end_col] = frame_normalized

        frames = combined_frames

    # Create up-scaled window
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", frame_width * 3, frame_height * 3)

    while True:
        for i, frame in enumerate(frames):
            if timestamps:
                frame_with_timestamp = frame.copy()
                timestamp = f"{timestamps[i]:.2f} s"
                cv2.putText(frame_with_timestamp,
                            timestamp, (5, frame_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1)
                displaying_frame = frame_with_timestamp
            else:
                displaying_frame = frame

            # Display the frame with OpenCV
            cv2.imshow("Frames", displaying_frame)

            # Wait for the frame_period duration and check for a key press
            key = cv2.waitKey(int(frame_period * 1000)) & 0xFF

            # If 'q' is pressed, exit the loop
            if key == ord('q'):
                cv2.destroyAllWindows()
                return


def save_frames(
    frames: np.ndarray,
    frame_rate: float,
    output_file: str or Path,
    active_channels: list,
    timestamps: bool,
    norm: list or tuple or np.ndarray = None
) -> None:
    """
    Save a sequence of imaging frames as a video file.

    This function takes a sequence of frames, optionally combines multiple
    channels into side-by-side visualizations, normalizes intensity values, 
    and saves the resulting video file. Timestamps can also be added to each 
    frame.

    Parameters
    ----------
    frames : np.ndarray
        A sequence of frames to save. For multi-channel input, frames should
        have shape (n_frames, n_channels, height, width).
    frame_rate : float
        The frame rate (in frames per second) for the output video.
    output_file : str | Path
        The path to the output video file. The directory must exist.
    active_channels : list, optional
        A list of channel indices to display. If specified, frames from these
        channels are combined side-by-side in the output. Default is None.
    timestamps : bool, optional
        If True, adds elapsed time annotations to each frame. Default is False.
    norm : list | tuple | np.ndarray, optional
        Normalization range for frame intensity values as [min, max]. 
        Default is [0, 255].

    Returns
    -------
    None
        Saves the video to the specified path.

    Raises
    ------
    FileNotFoundError
        If the parent directory of `output_file` does not exist.
    """

    output_path = Path(output_file)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Directory {output_path.parent}"
                                "does not exist.")

    vmin, vmax = norm if norm else (0, 255)

    if not active_channels:
        frame_height, frame_width = frames[0].shape
        n_frames = len(frames)
        
        # Pre-allocate normalized frames array for single channel
        normalized_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            normalized_frames[i] = cv2.normalize(
                frame, None, vmin, vmax, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frames = normalized_frames
    
    else:
        # Get dimensions for multi-channel combination
        n_frames, _, frame_height, single_width = frames.shape
        frame_width = single_width * len(active_channels)
        
        # Pre-allocate combined frames array
        combined_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.uint8)

        for i in range(n_frames):
            for n, channel in enumerate(active_channels):
                # Extract and normalize the channel frame
                channel_frame = frames[i, n, :, :]
                frame_normalized = cv2.normalize(
                    channel_frame, None, vmin, vmax, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Place in the correct position in combined frame
                start_col = n * single_width
                end_col = (n + 1) * single_width
                combined_frames[i, :, start_col:end_col] = frame_normalized

        frames = combined_frames

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        float(frame_rate),
        (frame_width, frame_height),
        isColor=False)

    if timestamps:
        n_frames = len(frames)
        timestamps = np.linspace(0, n_frames / frame_rate, n_frames)

    for i, frame in enumerate(frames):
        if timestamps:
            frame_with_timestamp = frame.copy()
            timestamp = f"{timestamps[i]:.2f} s"
            cv2.putText(
                frame_with_timestamp,
                timestamp, (5, frame_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 255, 255), 1)
            writer.write(frame_with_timestamp)
        else:
            writer.write(frame)

    # Release the video writer
    writer.release()


def save_single_frame(
    frame: np.ndarray,
    output_file: str,
    data_type: str,
    norm: tuple or list or np.ndarray = None
) -> None:
    """
    Save a single frame as a normalized image file.

    This function saves a single frame as an image file after applying
    optional normalization and converting the frame to the specified data type.

    Parameters
    ----------
    frame : np.ndarray
        The frame data to save, typically a 2D NumPy array.
    output_file : str
        Path to save the output image. The parent directory must exist.
    data_type : str
        Target data type for the saved frame. Supported types are:
        - 'int16': Signed 16-bit integer.
        - 'uint16': Unsigned 16-bit integer.
        - 'int8': Signed 8-bit integer.
    norm : tuple | list | np.ndarray, optional
        Normalization range as (min, max). If None, the range is derived from
        the frame's minimum and maximum values.

    Returns
    -------
    None
        The function writes the image to the specified file path.

    Raises
    ------
    FileNotFoundError
        If the parent directory of `output_file` does not exist.
    ValueError
        If an unsupported `data_type` is provided.

    Notes
    -----
    - For 'int16' and 'uint16' data types, the frame is normalized to the 
      specified range or to its original range if `norm` is not provided.
    - For 'uint16', bitwise XOR is applied to convert the signed data to unsigned.
    - For 'int8', the frame is normalized using OpenCV's `cv2.normalize` function.
    - The output is saved using OpenCV's `cv2.imwrite`.
    """

    output_path = Path(output_file)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Directory {output_path.parent}"
                                "does not exist.")
    
    if data_type not in ['int16', 'uint16', 'int8']:
        raise ValueError(
            f"Unsupported data_type: {data_type}. "
            "Supported types are 'int16', 'uint16', 'int8'.")

    # Default scanimage data type
    if data_type == 'int16':

        vmin, vmax = norm if norm else (frame.min(), frame.max())

        frame = frame.astype(np.float64)
        frame_normalized = (
            (frame - frame.min()) /
            (frame.max() - frame.min()))
        frame_normalized = (
            frame_normalized * (vmax - vmin) + vmin
        ).astype(np.int16)

    elif data_type == 'uint16':

        frame = np.bitwise_xor(frame.view(np.uint16), np.uint16(32768))
        vmin, vmax = norm if norm else (frame.min(), frame.max())

        frame_normalized = (
            (frame - frame.min()) /
            (frame.max() - frame.min()))
        frame_normalized = (
            frame_normalized * (vmax - vmin) + vmin
        ).astype(np.uint16)

    elif data_type == 'int8':

        vmin, vmax = norm if norm else (0, 255)
        frame_normalized = cv2.normalize(
            frame,
            None,
            vmin,
            vmax,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U)

    cv2.imwrite(output_file, frame_normalized)


def save_processed_arrays(
        data: list,
        save_path: str or Path,
        processing_params: dict,
        overwrite: bool = False
) -> Path:
    """
    Save processed imaging data arrays to disk using HDF5 format.

    This function saves a list of processed numpy arrays (typically from
    ImagingDataset._load_data()) along with the processing parameters used.
    This allows for fast loading of preprocessed data in subsequent sessions.

    Parameters
    ----------
    data : List[np.ndarray]
        List of processed imaging arrays, typically one per sweep group.
    save_path : str or Path
        Path where to save the processed data file.
    processing_params : Dict[str, Any]
        Dictionary containing processing parameters used, such as:
        - 'kernel_size_um': tuple of filter kernel sizes
        - 'pmt_artifact_detection_threshold': PMT threshold value
        - Any other relevant processing parameters
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If data is empty or invalid.

    Example
    -------
    >>> # After processing data normally
    >>> dataset = ImagingDataset(folder)  # This processes the data
    >>>
    >>> # Save for future use
    >>> processing_params = {
    ...     'kernel_size_um': dataset.median_filter_kernel_size_um,
    ...     'pmt_threshold': dataset.pmt_artifact_detection_threshold
    ... }
    >>> save_path = dataset.folder / "processed_data.h5"
    >>> save_processed_arrays(dataset.data, save_path, processing_params)
    """
    save_path = Path(save_path)

    if not data:
        raise ValueError("Data list is empty - nothing to save")

    if save_path.exists() and not overwrite:
        print(
            f"File already exists: {save_path}. "
            "Use overwrite=True to replace.")
        return

    # Calculate metadata
    total_size_mb = sum(arr.nbytes for arr in data) / (1024**2)
    array_info = []
    for i, arr in enumerate(data):
        shape = arr.shape
        array_info.append({
            'roi_index': i,
            'shape': shape,
            'dtype': str(arr.dtype),
            'size_mb': arr.nbytes / (1024**2),
            'n_frames': shape[0] if len(shape) >= 1 else 'unknown',
            'n_channels': shape[1] if len(shape) >= 2 else 'unknown',
            'height': shape[2] if len(shape) >= 3 else 'unknown',
            'width': shape[3] if len(shape) >= 4 else 'unknown'
        })

    # Prepare data structure for saving
    save_data = {
        'processed_arrays': data,
        'processing_params': processing_params,
        'metadata': {
            'version': '1.0',
            'timestamp': time.time(),
            'n_arrays': len(data),
            'array_info': array_info,
            'total_size_mb': total_size_mb,
            'saved_with': 'spyne.core.imaging.io.save_processed_arrays'
        }
    }

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save using flammkuchen (HDF5)
        fl.save(save_path, save_data)

        file_size_mb = save_path.stat().st_size / (1024**2)
        print(f"✓ Saved processed arrays ({file_size_mb:.1f} MB) "
              f"to {save_path}")

        return

    except Exception as e:
        # Clean up partial file on error
        if save_path.exists():
            save_path.unlink()
        raise RuntimeError(f"Failed to save processed arrays: {e}") from e


def load_processed_arrays(
        load_path: str or Path
) -> tuple:
    """
    Load processed imaging data arrays from disk.

    Parameters
    ----------
    load_path : str or Path
        Path to the saved processed arrays file.

    Returns
    -------
    Tuple[List[np.ndarray], Dict[str, Any]]
        Tuple containing:
        - List of processed imaging arrays
        - Dictionary with processing parameters and metadata

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the file format is invalid or corrupted.

    Example
    -------
    >>> # Load previously saved processed data
    >>> load_path = Path("path/to/processed_data.h5")
    >>> data, info = load_processed_arrays(load_path)
    >>>
    >>> # Check what processing was used
    >>> print(f"Kernel size: {info['processing_params']['kernel_size_um']}")
    >>> print(f"Arrays loaded: {len(data)}")
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(
            f"Processed arrays file not found: {load_path}")

    start_time = time.time()

    try:
        # Load data using flammkuchen
        saved_data = fl.load(load_path)

        # Extract components
        arrays = saved_data['processed_arrays']
        processing_params = saved_data['processing_params']
        metadata = saved_data.get('metadata', {})

        # Validate data
        if not arrays:
            raise ValueError("No arrays found in saved file")

        elapsed = time.time() - start_time
        file_size_mb = load_path.stat().st_size / (1024**2)
        n_arrays = len(arrays)

        print(f"✓ Loaded {n_arrays} processed arrays from: {load_path}")

        # Return arrays and combined info
        info = {
            'processing_params': processing_params,
            'metadata': metadata,
            'load_time_seconds': elapsed,
            'file_size_mb': file_size_mb
        }

        return arrays, info

    except Exception as e:
        raise ValueError(
            f"Failed to load processed arrays from {load_path}: {e}") from e


def save_metadata(
    metadata: dict,
    save_path: str or Path,
    overwrite: bool = False
) -> None:
    """
    Save the metadata to a .json file for fast loading and inspection.

    Parameters
    ----------
    save_path : Path, optional
        Path where to save the metadata. If None, uses 'metadata.json' in the dataset folder.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if metadata is None:
        raise ValueError("No metadata to save. Load metadata first.")

    save_path = Path("metadata.json") if save_path is None else Path(save_path)

    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists. Use overwrite=True to overwrite.")

    # Save metadata using flammkuchen (HDF5)
    fl.save(save_path, metadata)
    file_size_mb = save_path.stat().st_size / (1024**2)
    print(f"✓ Saved metadata ({file_size_mb:.2f} MB) to {save_path}.")
    return


def load_saved_metadata(
    load_path: str or Path
) -> dict:
    """
    Load metadata from a saved .json file.

    Parameters
    ----------
    load_path : str or Path
        Path to the saved metadata file.

    Returns
    -------
    dict
        The loaded metadata dictionary.
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {load_path}")
    
    metadata = fl.load(load_path)
    
    print(f"✓ Loaded metadata from {load_path}.")
    
    return metadata