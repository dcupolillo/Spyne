from tqdm import tqdm
import tifffile
import numpy as np
from skimage.util import img_as_uint
import skimage
from scipy.ndimage import median_filter
from spyne.core.utils.pyabf_adc import get_digital_output_list
from spyne.core.utils.movie_utils import rows_deviation, modify_frames


def load_metadata_from_tiff(
        dataset_instance: object,
) -> tuple:

    metadata = []

    rect_periods = [
        rect.rectangle_period
        for zplane in dataset_instance.sf.neuComp
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
        for zplane in dataset_instance.sf.neuComp
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
                # BUG: there is sometimes uncorrespondence between
                # the metadata saved during data acquisition
                # and generated ROIs in ROIpy. Probably due to ROIpy 
                # updating following data collection (.roi files generated
                # at a previous time).
                try:
                    roi_in_scanfield_object = (
                        dataset_instance.sf.neuComp[file_n][n_roi_in_z])
                except:
                    print(file_n, file_path)
                    break

                z = float(roi['name'].split(",")[0].split(" = ")[-1])

                # Further access nested metadata
                scanfield = roi['scanfields']
                center_xy = scanfield['centerXY']
                size_xy = scanfield['sizeXY']
                pixel_resolution_xy = scanfield['pixelResolutionXY']

                center_xy_ref = np.dot(
                    [0.5, 0.5],
                    np.array(scanfield['affine']).T[:-1, :])

                translate = np.array(
                    [[1, 0, center_xy[0] - center_xy_ref[0]],
                        [0, 1, center_xy[1] - center_xy_ref[1]],
                        [0, 0, 1]])

                resolution = [
                    size_xy[0] / pixel_resolution_xy[0],
                    size_xy[1] / pixel_resolution_xy[1]]

                if z not in coplanar_dict:
                    coplanar_dict[z] = []
                coplanar_dict[z].append(n_roi_overall)

                # Take ROIpy.Scanfields.Roi specific metadata
                branch_degree = roi_in_scanfield_object.branch_degree
                branch_id = roi_in_scanfield_object.branch_id
                
                # NOTE: I'd rather take this info from ROIpy.Scanfields
                # objects for correspondence reasons, as after data
                # acquisition, ScanImage applies new UUIDs to ROIs
                roi_uuid = roi_in_scanfield_object.roi_uuid
                roi_uuid_uint64 = roi_in_scanfield_object.roi_uuid_uint64
                
                # Write the metadata entry
                metadata.append({
                    'objective resolution':
                        frame_data["SI.objectiveResolution"],
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
                    'size_xy': scanfield['sizeXY'],
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
                    'median_filter_kernel_size': (
                        dataset_instance.median_filter_kernel_size),
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

    output_metadata = [meta
                       for group in grouped_metadata
                       for meta in group]

    return n_roi_overalls, output_metadata


def load_imaging_data_from_tiff(
        dataset_instance: object,
        threshold_factor: int = 3,
) -> np.ndarray:

    data = [None] * len(dataset_instance.file_list)

    for n, file_path in enumerate(tqdm(
            dataset_instance.file_list, desc="Loading data")):

        # raw data are dtype signed int16
        raw_frames = tifffile.imread(file_path)

        # Negative values clipped as 0
        # FIXME: fix the range
        frames = raw_frames.copy()
        # frames = img_as_uint(frames)

        # Correct gating-PMT black stripe
        # Dectect deviating rows
        deviation_channel1 = rows_deviation(
            frames[:, 0, :, :], threshold_factor=threshold_factor, plot=False)
        deviation_channel2 = rows_deviation(
            frames[:, 1, :, :], threshold_factor=threshold_factor, plot=False)

        deviating_rows = {}

        for frame in set(deviation_channel1.keys()).union(
                set(deviation_channel2.keys())):

            rows1 = deviation_channel1.get(frame, [])
            rows2 = deviation_channel2.get(frame, [])
            deviating_rows[frame] = sorted(set(rows1 + rows2))

        sorted_deviating_rows = dict(sorted(deviating_rows.items()))

        consecutive_deviating_rows = {}
        frame_numbers = list(sorted_deviating_rows.keys())

        for frame_number in frame_numbers:
            consecutive_deviating_rows[frame_number] = (
                sorted_deviating_rows[frame_number])

        modified_frames = modify_frames(
            frames, consecutive_deviating_rows)

        # Apply 3D median filter
        filtered_frames = modified_frames.copy()
        filtered_frames[:, 0, :, :] = median_filter(
            filtered_frames[:, 0, :, :],
            size=dataset_instance.median_filter_kernel_size,
            mode='wrap')
        filtered_frames[:, 1, :, :] = median_filter(
            filtered_frames[:, 1, :, :],
            size=dataset_instance.median_filter_kernel_size,
            mode='wrap')

        data[n] = modified_frames

    return data
