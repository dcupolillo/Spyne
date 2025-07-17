from tqdm import tqdm
import tifffile
import numpy as np
from skimage.util import img_as_uint
from scipy.ndimage import median_filter
from spyne.core.utils.pyabf_adc import get_digital_output_list
from spyne.core.utils.movie_utils import rows_deviation, modify_frames
from spyne.core.utils.denoise import radius_to_kernel_size


def load_metadata_from_tiff(
        dataset_instance: object,
) -> tuple:

    metadata = []

    rect_periods = [
        rect.rectangle_period
        for zplane in dataset_instance._sf.neuron
        for rect in zplane]

    unique_roifile_list = list(
        {file_path.parent: file_path
            for file_path in dataset_instance.file_list}.values())

    z_values = sorted(
        {int(file_path.parent.name[1:])
            for file_path in dataset_instance.file_list})
    z_index_map = {z: idx for idx, z in enumerate(z_values)}

    grouped_sweeps = [[] for _ in range(len(z_values))]

    for file_path in dataset_instance.file_list:

        z_value = int(file_path.parent.name[1:])
        z_index = z_index_map[z_value]
        grouped_sweeps[z_index].append(file_path)

    dataset_instance.n_sweeps = [len(n) for n in grouped_sweeps]

    dataset_instance.coplanar_n = [
        rect.z_ind
        for zplane in dataset_instance._sf.neuron
        for rect in zplane]

    # Retrieve metadata
    n_roi_overall = 0
    coplanar_dict = {}

    for file_n, file_path in enumerate(
            tqdm(unique_roifile_list, desc="Loading metadata")):

        abf_file = dataset_instance.abf_file_list[file_n]
        adc_list = get_digital_output_list(abf_file)

        with tifffile.TiffFile(file_path) as tif:

            scanimage_metadata = tif.scanimage_metadata

            frame_data = scanimage_metadata['FrameData']
            roigroup_data = (
                scanimage_metadata['RoiGroups']['imagingRoiGroup'])
            rois = roigroup_data['rois']

            rois_list = []
            if isinstance(rois, dict):
                rois_list.append(rois)
            elif isinstance(rois, list):
                rois_list = rois
            else:
                raise TypeError(f"Unexpected type for rois: {type(rois)}")

            for n_roi_in_z, roi in enumerate(rois_list):

                # sanity check for single roi extracted metadata
                if not isinstance(roi, dict):
                    raise TypeError("Error in metadata type.")

                # Call the corresponding ROIpy.Scanfields.Roi object
                # NOTE: there is uncorrespondence between
                # the metadata saved during data acquisition
                # and generated ROIs in ROIpy. Probably due to ROIpy
                # updating following data collection (.roi files generated
                # at a previous time). This is handled by using ROIpy UUIDs
                # instead of ScanImage UUIDs (see lines 141-142).

                try:
                    roi_in_scanfield_object = (
                        dataset_instance._sf.neuron[file_n][n_roi_in_z])
                except:
                    print(file_n, file_path)
                    break

                z = float(roi['name'].split(",")[0].split(" = ")[-1])

                objective_resolution = frame_data["SI.objectiveResolution"]

                scanfield = roi['scanfields']
                center_xy = scanfield['centerXY']
                size_xy_degree = scanfield['sizeXY']
                size_xy_um = [
                    size * objective_resolution
                    for size in scanfield['sizeXY']]
                pixel_resolution_xy = scanfield['pixelResolutionXY']

                center_xy_ref = np.dot(
                    [0.5, 0.5],
                    np.array(scanfield['affine']).T[:-1, :])

                translate = np.array(
                    [[1, 0, center_xy[0] - center_xy_ref[0]],
                        [0, 1, center_xy[1] - center_xy_ref[1]],
                        [0, 0, 1]])

                resolution = [
                    size_xy_um[0] / pixel_resolution_xy[0],
                    size_xy_um[1] / pixel_resolution_xy[1]]

                # Calculate pixel kernel size from micrometers
                kernel_size_um = dataset_instance.median_filter_kernel_size_um
                spatial_radius_um = kernel_size_um[0]  # Assume square kernel
                spatial_kernel_size = radius_to_kernel_size(
                    spatial_radius_um,
                    resolution,
                    ensure_odd=True,
                    min_size=3
                )
                temporal_kernel_size = int(kernel_size_um[2])
                kernel_size_pixels = (
                    temporal_kernel_size,
                    spatial_kernel_size[1],  # height (y) dimension
                    spatial_kernel_size[0])  # width (x) dimension

                if z not in coplanar_dict:
                    coplanar_dict[z] = []
                coplanar_dict[z].append(n_roi_overall)

                # Take ROIpy.Scanfields.Roi specific metadata
                branch_degree = roi_in_scanfield_object.branch_degree
                branch_id = roi_in_scanfield_object.branch_id

                # Use ROIpy UUIDs instead of ScanImage UUIDs
                # for better correspondence
                roi_uuid = roi_in_scanfield_object.roi_uuid
                roi_uuid_uint64 = roi_in_scanfield_object.roi_uuid_uint64

                # Write the metadata entry
                metadata.append({
                    'objective resolution': objective_resolution,
                    'affine': scanfield['affine'],
                    'translate': translate,
                    'center_xy': center_xy,
                    'center_xy_ref': center_xy_ref,
                    'pixel_resolution_xy': pixel_resolution_xy,
                    'pixel_to_ref_transform':
                        scanfield['pixelToRefTransform'],
                    'roi_uuid': roi_uuid,
                    'roi_uuid_int64': roi_uuid_uint64,
                    'rotation_degrees': scanfield['rotationDegrees'],
                    'size_xy_degree': size_xy_degree,
                    'size_xy_um': size_xy_um,
                    'resolution': resolution,
                    'z': z,
                    'z_ind': file_n,
                    'n_roi': n_roi_overall,
                    'branch_degree': branch_degree,
                    'branch_id': branch_id,
                    'n_sweeps': dataset_instance.n_sweeps[file_n],
                    'n_frames':
                        frame_data['SI.hStackManager.framesPerSlice'],
                    'frame_rate':
                        frame_data['SI.hRoiManager.scanFrameRate'],
                    'rectangle_period': rect_periods[n_roi_overall],
                    'flyto':
                        frame_data['SI.hScan2D.flytoTimePerScanfield'],
                    'flyback':
                        frame_data['SI.hScan2D.flybackTimePerFrame'],
                    'adc_list': [adc for adc in adc_list],
                    'ch_active': frame_data['SI.hChannels.channelsActive'],
                    'LUT': frame_data['SI.hChannels.channelLUT'],
                    'offset': frame_data['SI.hChannels.channelOffset'],
                    'subtract_offset':
                        frame_data['SI.hChannels.channelSubtractOffset'],
                    'input_range':
                        frame_data['SI.hChannels.channelInputRange'],
                    'median_filter_kernel_size_um': (
                        dataset_instance.median_filter_kernel_size_um),
                    'median_filter_kernel_size_px': kernel_size_pixels,
                })

                n_roi_overall += 1

    for meta in metadata:
        z = meta['z']
        meta['coplanar_roi_n'] = coplanar_dict[z]

    grouped_metadata = [[] for _ in range(len(z_values))]

    for meta in metadata:
        grouped_metadata[meta['z_ind']].append(meta)

    n_roi_overalls = len(metadata)

    for group in grouped_metadata:
        start_y = 0
        for meta in group:
            end_y = start_y + meta['pixel_resolution_xy'][1]
            meta['roi_bounds'] = [start_y, end_y]
            start_y = end_y

    output_metadata = [
        meta
        for group in grouped_metadata
        for meta in group]

    return n_roi_overalls, output_metadata


def load_imaging_data_from_tiff(
        dataset_instance: object,
        threshold_factor: int = 3,
) -> np.ndarray:

    data = [None] * len(dataset_instance.file_list)

    # Create file to ROI index mapping once
    file_to_roi_map = create_file_to_roi_map(dataset_instance)

    for n, file_path in enumerate(tqdm(
            dataset_instance.file_list, desc="Loading data")):

        # raw data are dtype signed int16
        raw_frames = tifffile.imread(file_path)

        # Negative values clipped as 0
        # FIXME: fix the range
        frames = raw_frames.copy()
        # frames = img_as_uint(frames)

        # Get metadata for this file using the pre-computed mapping
        roi_index = file_to_roi_map[file_path]
        file_metadata = dataset_instance.metadata[roi_index]

        # Get active channels from metadata
        # and convert to 0-indexed
        active_channels = [ch - 1 for ch in file_metadata['ch_active']]

        # Correct gating-PMT black stripe
        # Detect deviating rows for each active channel
        channel_deviations = {}
        for channel_idx in active_channels:
            channel_deviations[channel_idx] = rows_deviation(
                frames[:, channel_idx, :, :],
                threshold_factor=threshold_factor,
                plot=False)

        # Combine deviating rows from all active channels
        deviating_rows = {}
        all_frames = set()
        for channel_deviation in channel_deviations.values():
            all_frames.update(channel_deviation.keys())

        for frame in all_frames:
            rows_for_frame = []
            for channel_deviation in channel_deviations.values():
                rows_for_frame.extend(channel_deviation.get(frame, []))
            deviating_rows[frame] = sorted(set(rows_for_frame))

        sorted_deviating_rows = dict(sorted(deviating_rows.items()))

        consecutive_deviating_rows = {}
        frame_numbers = list(sorted_deviating_rows.keys())

        for frame_number in frame_numbers:
            consecutive_deviating_rows[frame_number] = (
                sorted_deviating_rows[frame_number])

        modified_frames = modify_frames(
            frames, consecutive_deviating_rows)

        # Use pre-calculated pixel kernel size from metadata
        kernel_size_pixels = file_metadata['median_filter_kernel_size_px']

        # Apply 3D median filter to each active channel
        filtered_frames = modified_frames.copy()
        for channel_idx in active_channels:
            filtered_frames[:, channel_idx, :, :] = median_filter(
                filtered_frames[:, channel_idx, :, :],
                size=kernel_size_pixels,
                mode='wrap')

        data[n] = filtered_frames

    return data


def create_file_to_roi_map(dataset_instance: object) -> dict:
    """
    Create a mapping from file paths to ROI indices.

    Args:
        dataset_instance: The dataset instance containing
        file_list and metadata

    Returns:
        dict: Mapping from file paths to ROI indices for metadata lookup
    """
    # Get z-values and create index mapping
    z_values = sorted(
        {int(file_path.parent.name[1:])
         for file_path in dataset_instance.file_list})
    z_index_map = {z: idx for idx, z in enumerate(z_values)}

    # Create file to ROI index mapping
    file_to_roi_map = {}
    for file_path in dataset_instance.file_list:
        z_value = int(file_path.parent.name[1:])
        z_index = z_index_map[z_value]

        # Find the ROI index for this z-index
        roi_index = None
        for idx, meta in enumerate(dataset_instance.metadata):
            if meta['z_ind'] == z_index:
                roi_index = idx
                break

        if roi_index is None:
            raise ValueError(
                f"No metadata found for z-index {z_index}",
                "(z-value {z_value})")

        file_to_roi_map[file_path] = roi_index

    return file_to_roi_map
