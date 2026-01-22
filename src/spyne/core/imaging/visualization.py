""" Created on Wed Sep  6 11:31:57 2023
    @author: dcupolillo 
    
    Visualization and export functions for imaging data.
    
    This module handles frame visualization, video creation, and image export.
    Data persistence functions have been moved to dataloader.py for better
    separation of concerns.
    """

from pathlib import Path
import cv2
import numpy as np


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