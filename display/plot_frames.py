""" Created on Wed Sep  6 11:31:57 2023
    @author: dcupolillo """


from pathlib import Path
import cv2
import numpy as np


def animate_frames(
        frames: np.ndarray,
        frame_rate: float,
        active_channels: list = None,
        timestamps: bool = False,
        norm: list or tuple = None
) -> None:
    """
    Shows frame sequence in a loop.

    Keyword args:
        frames (list): input sequence of frames
        frame_rate (float): rate of frame display

    Optional:
        active_channels (list): if specified,
                                channels are displayed side by side
        timestamps (bool): if specified, elapsed time will appear on video

    """

    n_frames = len(frames)
    frame_period = 1 / frame_rate
    scan_duration = frame_period * n_frames

    if timestamps:
        timestamps = np.linspace(0, scan_duration, n_frames)

    norm_values = norm if norm else (0, 255)

    if not active_channels:
        frame_height, frame_width = frames[0].shape
    else:
        combined_frames = []

        for i, frame in enumerate(frames):
            combined_channels = []

            for n, channel in enumerate(active_channels):
                # Extract the channel frame
                channel_frame = frames[i, n, :, :]
                frame_normalized = cv2.normalize(
                    channel_frame,
                    None,
                    norm_values[0], norm_values[1],
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U)

                combined_channels.append(frame_normalized)

            combined_frame = np.column_stack(combined_channels)
            combined_frames.append(combined_frame)

        frames = np.array(combined_frames)
        frame_height, frame_width = frames[0].shape

    # Create up-scaled window
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", frame_width * 3, frame_height * 3)

    while True:
        for i, frame in enumerate(frames):
            frame_normalized = cv2.normalize(
                frame,
                None,
                norm_values[0], norm_values[1],
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U)

            if timestamps:
                frame_with_timestamp = frame_normalized.copy()
                timestamp = f"{timestamps[i]:.2f} s"
                cv2.putText(frame_with_timestamp,
                            timestamp, (5, frame_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1)
                displaying_frame = frame_with_timestamp
            else:
                displaying_frame = frame_normalized

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
        active_channels: list = None,
        timestamps: bool = False,
        norm: list or tuple or np.ndarray = None
) -> None:
    """
    Saves frame sequence as a video.

    Keyword args:
        frames (list): input sequence of frames
        frame_rate (float): rate of frame display
        output_file (str): path to saved video

    Optional:
        active_channels (list): if specified,
                                channels are displayed side by side
        timestamps (bool): if specified, elapsed time will appear on video

    """

    output_path = Path(output_file)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Directory {output_path.parent}"
                                "does not exist.")

    norm_values = norm if norm else (0, 255)

    if not active_channels:
        frame_height, frame_width = frames[0].shape
    else:
        combined_frames = []

        for i, frame in enumerate(frames):
            combined_channels = []

            for n, channel in enumerate(active_channels):
                # Extract the channel frame
                channel_frame = frames[i, n, :, :]
                frame_normalized = cv2.normalize(
                    channel_frame,
                    None,
                    norm_values[0], norm_values[1],
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U)
                combined_channels.append(frame_normalized)

            combined_frame = np.column_stack(combined_channels)
            combined_frames.append(combined_frame)

        frames = np.array(combined_frames)
        frame_height, frame_width = frames[0].shape

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
        frame_normalized = cv2.normalize(
            frame,
            None,
            norm_values[0], norm_values[1],
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U)

        if timestamps:
            frame_with_timestamp = frame_normalized.copy()
            timestamp = f"{timestamps[i]:.2f} s"
            cv2.putText(frame_with_timestamp,
                        timestamp, (5, frame_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1)
            writer.write(frame_with_timestamp)
        else:
            writer.write(frame_normalized)

    # Release the video writer
    writer.release()


def save_single_frame(
        frame: np.ndarray,
        output_file: str,
        data_type: str,
        norm: tuple or list or np.ndarray = None
) -> None:

    output_path = Path(output_file)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Directory {output_path.parent}"
                                "does not exist.")

    # Default scanimage data type
    if data_type == 'int16':

        norm_values = norm if norm else (frame.min(), frame.max())

        frame = frame.astype(np.float64)
        frame_normalized = ((frame - frame.min()) /
                            (frame.max() - frame.min()))
        frame_normalized = (frame_normalized *
                            (norm_values[1] - norm_values[0])
                            + norm_values[0]).astype(np.int16)

    elif data_type == 'uint16':

        frame = np.bitwise_xor(frame.view(np.uint16), np.uint16(32768))
        norm_values = norm if norm else (frame.min(), frame.max())

        frame_normalized = ((frame - frame.min()) /
                            (frame.max() - frame.min()))
        frame_normalized = (frame_normalized *
                            (norm_values[1] - norm_values[0])
                            + norm_values[0]).astype(np.uint16)

    elif data_type == 'int8':

        frame_normalized = cv2.normalize(
            frame,
            None,
            norm_values[0],
            norm_values[1],
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U)

    else:
        raise ValueError('Invalid data_type')

    cv2.imwrite(output_file, frame_normalized)


# def plot_sf(frame, ax=None, ):

#     if not ax:
#         fig, ax = plt.subplots()
#         ax.set_aspect('equal')
#         ax.invert_yaxis()

#     if type(frame).__name__ == 'Roi':  # if it is single roi

#         # frame is a Singlechannel object
#         frame_normalized = cv2.normalize(frame.ch2.maxproj.frame,
#                                          None, 0, 255,
#                                          cv2.NORM_MINMAX,
#                                          dtype=cv2.CV_8U)

#         pixelresolutionXY = frame_normalized.shape
#         centerXY = frame.roi_center

#         pixToRefT = np.array(frame.roi_pixelToRef)
#         sfToRefT = np.array(frame.roi_affine)

#         # Create a meshgrid of frame pixels coordinates
#         # and convert to list of [x, y] coordinates of n pixels
#         x_coords = np.arange(pixelresolutionXY[1])
#         y_coords = np.arange(pixelresolutionXY[0])
#         meshgrid_ref = np.array(np.meshgrid(
#             x_coords, y_coords)).T.reshape(-1, 2)  # list
#         # of coordinates
#         # of all pixels

#         # Get intensity values as a 1D list
#         intensity_values = frame_normalized.T.flatten()

#         # Transform the meshgrid points
#         pts_in_scanfield = transform(meshgrid_ref,
#                                      sfToRefT, pixToRefT,
#                                      centerXY, pixelresolutionXY)

#         markersize = 40
#         ax.scatter(pts_in_scanfield[:, 0], pts_in_scanfield[:, 1],
#                    c=intensity_values, cmap='viridis',
#                    s=markersize, marker='s')

#     elif type(frame).__name__ == 'Movie':

#         for roi in frame.roiList:

#             # frame is a Singlechannel object
#             frame_normalized = cv2.normalize(frame[roi].ch2.maxproj.frame,
#                                              None, 0, 255,
#                                              cv2.NORM_MINMAX,
#                                              dtype=cv2.CV_8U)

#             pixelresolutionXY = frame_normalized.shape
#             centerXY = frame[roi].roi_center

#             pixToRefT = np.array(frame[roi].roi_pixelToRef)
#             sfToRefT = np.array(frame[roi].roi_affine)

#             # Create a meshgrid of frame pixels coordinates
#             # and convert to list of [x, y] coordinates of n pixels
#             x_coords = np.arange(pixelresolutionXY[1])
#             y_coords = np.arange(pixelresolutionXY[0])
#             meshgrid_ref = np.array(np.meshgrid(
#                 x_coords, y_coords)).T.reshape(-1, 2)

#             # Get intensity values as a 1D list
#             intensity_values = frame_normalized.T.flatten()

#             # Transform the meshgrid points
#             pts_in_scanfield = transform(meshgrid_ref,
#                                          sfToRefT, pixToRefT,
#                                          centerXY, pixelresolutionXY)

#             markersize = 40
#             ax.scatter(pts_in_scanfield[:, 0], pts_in_scanfield[:, 1],
#                        c=intensity_values, cmap='binary_r',
#                        s=markersize, marker='s')
