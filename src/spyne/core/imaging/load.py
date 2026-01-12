""" Created on Fri Aug  2 14:58:52 2024
    @author: dcupolillo

    Loading logics for metadata and data from raw ScanImage TIFF files.
"""

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
from skimage.util import img_as_uint
from scipy.ndimage import median_filter
from spyne.core.electrophysiology.load import get_digital_output_list
from spyne.core.imaging.preprocessing import (
    modify_frames, collect_deviating_rows_for_channels, radius_to_kernel_size)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spyne.core.imaging.imagingdataset import ImagingDataset


def create_file_to_roi_map(
        dataset_instance: ImagingDataset,
) -> dict:
    """
    Create a mapping from file paths to ROI indices.

    Parameters
    ----------
    dataset_instance : ImagingDataset
        The dataset instance containing file_list and metadata.

    Returns
    -------
    dict
        Mapping from file paths to ROI indices for metadata lookup.
    """
    file_list = dataset_instance.file_list

    # Get z-values and create index mapping
    z_values = sorted(
        {int(file_path.parent.name[1:])
         for file_path in file_list})
    z_index_map = {z: idx for idx, z in enumerate(z_values)}

    # Create file to ROI index mapping
    file_to_roi_map = {}
    for file_path in file_list:
        z_value = int(file_path.parent.name[1:])
        z_index = z_index_map[z_value]

        # Find all ROI indices for this z-index
        roi_indices = []
        for idx, meta in enumerate(dataset_instance.metadata):
            if meta['z_ind'] == z_index:
                roi_indices.append(idx)

        if not roi_indices:
            raise ValueError(
                f"No metadata found for z-index {z_index}",
                "(z-value {z_value})")

        file_to_roi_map[file_path] = roi_indices

    return file_to_roi_map


def load_metadata_from_tiff(
        dataset_instance: ImagingDataset,
) -> tuple:
    """
    Extract and organize metadata from ScanImage TIFF files.

    This function processes TIFF files from ScanImage microscopy data,
    extracting comprehensive metadata including ROI information,
    scanning parameters, and coordinate transformations. It integrates
    with ROIpy scanfield objects to provide morphological annotations
    and handles coordinate system transformations between ScanImage
    and reference frames.

    Parameters
    ----------
    dataset_instance : ImagingDataset
        Dataset instance containing file_list, abf_file_list,
        _sf (scanfield), and median_filter_kernel_size_um attributes.

    Returns
    -------
    tuple
        A tuple containing:
        - n_roi_overalls (int): Total number of ROIs across all files
        - output_metadata (list[dict]): List of metadata dictionaries,
          each containing:
            - Coordinate transforms (affine, translate, pixel_to_ref)
            - ROI geometry (center_xy, size_xy_um, pixel_resolution_xy)
            - Scanning parameters (frame_rate, flyto, flyback)
            - Channel configuration (ch_active, LUT, offset)
            - ROIpy integration (roi_uuid, branch_degree, branch_id)
            - Filter parameters (median_filter_kernel_size_px)
            - Temporal info (n_frames, n_sweeps, rectangle_period)
            - Z-plane organization (z, z_ind, coplanar_roi_n, roi_bounds)

    Notes
    -----
    - Uses ROIpy UUIDs instead of ScanImage UUIDs for better correspondence
    - Converts spatial filter kernel sizes from micrometers to pixels
    - Organizes ROIs by z-plane and assigns vertical bounds for stacking
    - Handles both single ROI (dict) and multiple ROI (list) configurations
    """

    metadata = []

    rect_periods = [
        rect.rectangle_period
        for zplane in dataset_instance._sf.neuron
        for rect in zplane]

    # Take 1 file per z-plane for metadata extraction
    # to avoid redundant processing
    unique_roifile_list = list(
        {file_path.parent: file_path
            for file_path in dataset_instance.file_list}.values())

    # Z plane to index mapping
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

        # Extract z-value from file path to get correct z-index
        z_value = int(file_path.parent.name[1:])
        z_index = z_index_map[z_value]

        abf_file = dataset_instance.abf_file_list[file_n]
        adc_list = get_digital_output_list(abf_file)

        with tifffile.TiffFile(file_path) as tif:

            scanimage_metadata = tif.scanimage_metadata

            frame_data = scanimage_metadata['FrameData']
            roigroup_data = (
                scanimage_metadata['RoiGroups']['imagingRoiGroup'])
            rois = roigroup_data['rois']

            rois_list = rois if isinstance(rois, list) else [rois]

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
                spatial_radius_y_um = kernel_size_um[1]
                temporal_kernel_size = int(kernel_size_um[2])

                # Convert spatial radii to kernel sizes
                # radius_to_kernel_size takes radius and returns
                # (height, width)
                kernel_size_y, kernel_size_x = radius_to_kernel_size(
                    spatial_radius_y_um,
                    tuple(resolution),  # (x_resolution, y_resolution)
                    ensure_odd=True,
                    min_size=3
                )

                # Create a tuple for the kernel size in pixels
                # (temporal, height, width)
                kernel_size_pixels = (
                    temporal_kernel_size,
                    int(kernel_size_y),
                    int(kernel_size_x)
                )

                if z not in coplanar_dict:
                    coplanar_dict[z] = []
                coplanar_dict[z].append(n_roi_overall)

                # Take ROIpy.Scanfields.Roi specific metadata
                branch_degree = roi_in_scanfield_object.branch_degree
                branch_id = roi_in_scanfield_object.branch_id
                compartment = roi_in_scanfield_object.compartment

                # Use ROIpy UUIDs instead of ScanImage UUIDs
                # for better correspondence
                roi_uuid = roi_in_scanfield_object.roi_uuid
                roi_uuid_uint64 = roi_in_scanfield_object.roi_uuid_uint64

                # Write the metadata entry
                metadata.append(dict({
                    'dtype': frame_data["SI.hScan2D.channelsDataType"],
                    'objective_resolution': objective_resolution,
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
                    'compartment': compartment,
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
                    'pmt_artifact_detection_threshold':
                        dataset_instance.pmt_artifact_detection_threshold
                }))

                n_roi_overall += 1

    for meta in metadata:
        z = meta['z']
        meta['coplanar_roi_n'] = coplanar_dict[z]

    grouped_metadata = [[] for _ in range(len(z_values))]

    for meta in metadata:
        grouped_metadata[meta['z_ind']].append(meta)

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

    return n_roi_overall, output_metadata


def preprocess_tiff_file(
        file_path: Path,
        file_metadata: dict,
) -> np.ndarray:
    """
    Process a single TIFF file: load, correct artifacts, apply median filter.

    It processes each active channel independently and uses pre-calculated
    filter parameters from metadata for optimal performance.

    Parameters
    ----------
    file_path : Path
        Path to the raw TIFF file.
    file_metadata : dict
        Metadata for this file.

    Returns
    -------
    np.ndarray
        The processed 4D array for this file.

    Notes
    -----
    - Raw data expected as signed int16 from ScanImage
    - PMT black stripe correction applied per active channel
    - 3D median filtering with kernel size from metadata
    - Active channels converted from 1-indexed to 0-indexed
    - Filter kernel dimensions: (temporal, height, width)
    - Uses 'wrap' mode for boundary handling in median filter
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_frames = tifffile.imread(file_path)
    frames = raw_frames.copy()

    active_channels = [ch - 1 for ch in file_metadata['ch_active']]
    
    pmt_artifact_detection_threshold = file_metadata.get(
        'pmt_artifact_detection_threshold', 3)
    
    sorted_deviating_rows = collect_deviating_rows_for_channels(
        frames, active_channels,
        threshold_factor=pmt_artifact_detection_threshold
    )
    
    modified_frames = modify_frames(frames, sorted_deviating_rows)
    
    kernel_size_pixels = file_metadata['median_filter_kernel_size_px']

    processed_frames = modified_frames.copy()

    for channel_idx in active_channels:
        processed_frames[:, channel_idx, :, :] = median_filter(
            processed_frames[:, channel_idx, :, :],
            size=kernel_size_pixels,
            mode='wrap')

    return processed_frames


def load_imaging_data_from_tiff(
        dataset_instance: ImagingDataset,
) -> list:
    """
    Load and preprocess calcium imaging data from ScanImage TIFF files.

    This function relies on preprocess_tiff_file() to handle
    raw imaging data, PMT gating artifacts correction (black stripes)
    and 3D median filtering for noise reduction.

    Parameters
    ----------
    dataset_instance : ImagingDataset
        Dataset instance containing file_list, metadata, and 
        median_filter_kernel_size_um attributes.

    Returns
    -------
    list
        List of processed 4D arrays, one per file, with shape
        (n_frames, n_channels, height, width). Each array contains
        filtered calcium imaging data with PMT artifacts corrected.
    """

    data = [None] * len(dataset_instance.file_list)

    file_to_roi_map = dataset_instance.file_to_roi_map

    for n, file_path in enumerate(tqdm(
            dataset_instance.file_list, desc="Loading data")):
        
        roi_indices = file_to_roi_map[file_path]
        try:
            # Use the first ROI index
            # as relevant information for preprocessing,
            # such as median_filter_kernel_size_px,
            # are uniform across ROIs in the same file
            roi_index = roi_indices[0]  
            
            file_metadata = dataset_instance.metadata[roi_index]

            processed = preprocess_tiff_file(
                file_path=Path(file_path),
                file_metadata=file_metadata)
            
            # Ensure dtype is int16
            processed = processed.astype(np.int16)
            data[n] = processed
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return data
