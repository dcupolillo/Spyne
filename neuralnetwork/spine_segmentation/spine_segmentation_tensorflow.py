""" Created on Mon Oct 23 12:23:47 2023
    @author: dcupolillo """

from pathlib import Path
from typing import Union, Tuple, List
from collections import defaultdict

import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
# from skimage.measure import label, moments
# from skimage.measure import moments
# from scipy.ndimage import binary_dilation, binary_erosion
from skimage import morphology
# from scipy import ndimage
from tensorflow.keras.models import load_model
from spyne.core.utils.utils import tf_transform, generate_uuid


def pad_image(
        image: tf.Tensor,
        target_height: int,
        target_width: int
) -> tf.Tensor:
    """
    Helper function to pad a single
    image to the target height and width.
    """

    height, width = tf.shape(image)[0], tf.shape(image)[1]
    pad_height = target_height - height
    pad_width = target_width - width

    padded_image = tf.pad(
        image,
        paddings=[[0, pad_height], [0, pad_width]],
        mode='REFLECT')

    return padded_image


def pad_images(
        images: Union[List[tf.Tensor], tf.Tensor],
        pad_size: int = 32
) -> Union[List[tf.Tensor], Tuple[List[tf.Tensor], List[Tuple[int, int]]]]:
    """
    Pad a list of images or a single image to meet size requirements
    for the neural network model.

    Parameters
    ----------
    images : Union[List[tf.Tensor], tf.Tensor]
        The list of input images or a single image.
    pad_size : int, optional
        The size to pad to (default is 32).

    Returns
    -------
    Union[List[tf.Tensor], Tuple[List[tf.Tensor], List[Tuple[int, int]]]]
        The list of padded images or a single padded image.
    """

    if isinstance(images, list):
        # Find the maximum height and width
        max_height = max([tf.shape(image)[0] for image in images])
        max_width = max([tf.shape(image)[1] for image in images])

        # Ensure the dimensions are multiples of pad_size
        target_height = (max_height + pad_size - 1) // pad_size * pad_size
        target_width = (max_width + pad_size - 1) // pad_size * pad_size

        padded_images = [
            pad_image(image, target_height, target_width)
            for image in images]
        original_dimensions = [
            (tf.shape(image)[0].numpy(),
             tf.shape(image)[1].numpy())
            for image in images]

    else:
        # Single image case
        height, width = tf.shape(images)[0], tf.shape(images)[1]
        target_height = (height + pad_size - 1) // pad_size * pad_size
        target_width = (width + pad_size - 1) // pad_size * pad_size
        padded_images = pad_image(images, target_height, target_width)
        original_dimensions = (height, width)

    return padded_images, original_dimensions


def unpad_prediction(
        pred: tf.Tensor,
        original_dimensions: Tuple[int, int],
) -> tf.Tensor:
    """
    Helper function to unpad a single
    image to restore the original height and width.
    """
    original_height, original_width = original_dimensions

    return pred[:original_height, :original_width]


def unpad_images(
        predictions: Union[List[tf.Tensor], tf.Tensor],
        original_dimensions: Union[Tuple[int, int], List[Tuple[int, int]]]
) -> Union[List[tf.Tensor], tf.Tensor]:
    """
    Unpad a list of predictions or a single prediction based on the pad factor.

    Parameters
    ----------
    predictions : Union[List[tf.Tensor], tf.Tensor]
        The list of predictions or a single prediction.
    original_dimensions : Union[Tuple[int, int], List[Tuple[int, int]]]
        The original dimensions of the images before padding.

    Returns
    -------
    Union[List[tf.Tensor], tf.Tensor]
        The list of unpadded predictions or a single unpadded prediction.
    """
    if len(predictions.shape) != 2:
        unpadded_images = [
            unpad_prediction(pred, dims)
            for pred, dims in zip(predictions, original_dimensions)]
    else:
        unpadded_images = unpad_prediction(predictions, original_dimensions)

    return unpadded_images


def label(binary_image: tf.Tensor) -> tf.Tensor:
    """
    Perform connected component labeling on a binary image using TensorFlow.

    Parameters
    ----------
    binary_image : tf.Tensor
        The binary image to label.

    Returns
    -------
    tf.Tensor
        The labeled image.
    """
    binary_image = tf.cast(binary_image, tf.int32)

    # Pad the binary image to handle edge cases
    binary_image = tf.pad(
        binary_image,
        paddings=[[1, 1], [1, 1]],
        mode='CONSTANT',
        constant_values=0)

    # Get image dimensions
    shape = tf.shape(binary_image)

    # Initialize labels with zeros
    labels = tf.zeros(shape, dtype=tf.int32)
    label_counter = tf.Variable(1, dtype=tf.int32)

    # Create a grid of coordinates
    coords = tf.stack(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij'), axis=-1)

    # Flatten the binary image and coordinates
    flat_image = tf.reshape(binary_image, [-1])
    flat_coords = tf.reshape(coords, [-1, 2])

    # Get the indices of the foreground pixels
    foreground_indices = tf.where(flat_image > 0)
    foreground_coords = tf.gather(flat_coords, foreground_indices)

    # Convert to list of tuples
    foreground_coords = tf.reshape(foreground_coords, (-1, 2))

    # Iterate over the foreground pixels to assign labels
    for coord in tf.unstack(foreground_coords):
        x, y = coord[0], coord[1]

        # Check the 4-connected neighbors
        neighbors = tf.convert_to_tensor([
            [x-1, y],
            [x+1, y],
            [x, y-1],
            [x, y+1]
        ])

        assigned_label = 0
        for neighbor in tf.unstack(neighbors):
            nx, ny = neighbor[0], neighbor[1]
            if binary_image[nx, ny] > 0 and labels[nx, ny] > 0:
                assigned_label = labels[nx, ny]
                break

        if assigned_label > 0:
            labels = tf.tensor_scatter_nd_update(labels, [[x, y]], [assigned_label])
        else:
            labels = tf.tensor_scatter_nd_update(labels, [[x, y]], [label_counter])
            label_counter.assign_add(1)

    # Remove the padding
    labels = labels[1:-1, 1:-1]

    return labels


def inference(
        images: Union[tf.Tensor, List[tf.Tensor]],
        model_fn: Union[str, Path],
        original_dimensions: Union[Tuple[int, int], List[Tuple[int, int]]],
        pad_size=32,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load the model and run inference to get raw predictions.
    Handles both single images and batches: if images is a list,
    it is stacked into a 3D array (batch), whereas if images is a 2D array,
    it is expanded to a 3D array by adding a batch dimension.

    Parameters
    ----------
    images : Union[tf.Tensor, List[tf.Tensor]]
        The input images for segmentation. Can be a single image (2D array)
        or a batch of images (3D array or list of 2D arrays).
        Must be already padded.
    model_fn : Union[str, Path]
        Path to the pre-trained model.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Raw spine and dendrite predictions.
    """

    if images.ndim == 2:
        images = tf.expand_dims(images, axis=0)

    # Normalize the input image to have values between -1 and 1.
    images_min = tf.reduce_min(images, axis=(1, 2), keepdims=True)
    images_max = tf.reduce_max(images, axis=(1, 2), keepdims=True)

    with tf.device('/cpu:0'):
        images = (images - images_min) / (images_max - images_min) * 2 - 1

    # Load the model and run inference
    model_path = Path(__file__).resolve().parent.parent / model_fn
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Run inference on GPU
    with tf.device('/gpu:0'):
        model = load_model(model_path, compile=False)
        pd, ps = model.predict(images[..., None])

    ps = tf.squeeze(ps)
    pd = tf.squeeze(pd)

    spine_predictions = unpad_images(ps, original_dimensions)
    dendrite_predictions = unpad_images(pd, original_dimensions)

    return spine_predictions, dendrite_predictions


# TODO: convert to tensor!
def process_single_prediction(
        roi: object,
        spine_prediction: tf.Tensor,
        dendrite_prediction: tf.Tensor,
        spine_threshold: float,
        dendrite_threshold: float,
        min_spine_size: float,
        mask_size: int,
        min_distance: int,
        min_dendrite_size: float,
        dendrite_dilation_iterations: int,
        kernel_size: int,
        sd_factor: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Perform semantic segmentation to identify spines and dendrites.

    Parameters
    ----------
    roi : object (spyne.ImagingDataset.RoiGroup.Roi)
        Region of interest.
    base_image : np.ndarray
        The input image for segmentation.
    model_fn : str
        Path to the pre-trained model.
    spine_threshold : float
        Threshold value to identify spines.
    dendrite_threshold : float
        Threshold value to identify dendrites.
    min_spine_size : float
        Minimum size to keep spines.
    mask_size : int
        The size of the mask used for some operations.
    min_distance : int
        Minimum distance between detected centroids.
    min_dendrite_size : float
        Minimum size to keep dendrites.
    dendrite_dilation_iterations: int
        Number of iterations for dilating dendrites.
    kernel_size : int
        Size of the structuring element used for morphological operations.
    sd_factor: int
        Multiplier of SD to use as binary threshold.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        List of dictionaries containing spine information and
        dendrite information.
    """

    # Apply thresholds to generate binary masks
    spine_mask = threshold_prediction(
        spine_prediction, spine_threshold)
    dendrite_mask = threshold_prediction(
        dendrite_prediction, dendrite_threshold)

    # Labels same-value neighbors pixels
    # NOW WITH TENSORS
    spine_labels = label(spine_mask)
    dendrite_labels = label(dendrite_mask)

    # Remove corner joints from the spine mask
    spine_labels = remove_corner_joints(spine_labels)

    # Removes spine labels too distant from dendrite
    spine_labels = remove_distant_spines(
        spine_labels,
        dendrite_mask,
        dendrite_dilation_iterations)

    # Remove small labeled spines and dendrites
    # HINT : it might be an error to remove small spines?
    spine_labels = remove_small_labels(
        spine_labels,
        min_spine_size)
    dendrite_labels = remove_small_labels(
        dendrite_labels,
        min_dendrite_size)

    # Re-create binary mask from cleaned spine segmentation
    # spine_mask = np.where(spine_labels > 0, 255, 0).astype(np.uint8)
    # dendrite_mask = np.where(dendrite_labels > 0, 255, 0).astype(np.uint8)
    # Re-create binary mask from cleaned spine segmentation
    spine_mask = tf.where(spine_labels > 0, 255, 0)
    dendrite_mask = tf.where(dendrite_labels > 0, 255, 0)
    # Convert the tensors to numpy arrays
    # spine_mask_np = spine_mask.numpy().astype(np.uint8)
    # dendrite_mask_np = dendrite_mask.numpy().astype(np.uint8)

    # TODO: fill holes?

    # Create a distance map for each pixel to the nearest background pixel
    # cv2.DIST_L2 specifies Euclidean distance
    # distance_map = cv2.distanceTransform(
    #     spine_mask,
    #     cv2.DIST_L2,
    #     mask_size)

    # Use the tf custom function
    distance_map = tf_distance_transform(spine_mask)

    # local_maxima = peak_local_max(
    #     distance_map,
    #     footprint=np.ones((3, 3)),
    #     min_distance=min_distance,
    #     labels=spine_mask,
    #     exclude_border=False)

    def local_maxima_wrapper(image):
        # Convert TensorFlow tensor to Numpy array
        image_np = image.numpy()
        return peak_local_max(
            image_np,
            min_distance=min_distance,
            threshold_abs=spine_threshold)

    def watershed_wrapper(image, markers):
        image_np = image.numpy()
        markers_np = markers.numpy()
        return watershed(image_np, markers_np)

    local_maxima = tf.py_function(
        func=local_maxima_wrapper,
        inp=[distance_map],
        Tout=tf.int64)

    markers = tf.zeros_like(spine_mask, dtype=tf.uint8)

    # Place markers at the coordinates of the local maxima
    local_maxima = tf.reshape(local_maxima, [-1, 2])
    local_maxima = tf.cast(local_maxima, tf.int32)

    markers = tf.tensor_scatter_nd_update(
        markers,
        local_maxima,
        tf.ones(tf.shape(local_maxima)[0],
                dtype=tf.uint8)
        )

    # Apply watershed using tf.py_function
    watershed_spines = tf.py_function(
        func=watershed_wrapper,
        inp=[spine_mask, markers],
        Tout=tf.int64)

    # watershed_spines = tf.py_function(
    #     func=watershed,
    #     inp=[-distance_map,
    #          markers,
    #          spine_mask,
    #          np.ones((3, 3), dtype=bool),
    #          0.01],
    #     Tout=tf.int32
    # )

    # markers = np.zeros_like(spine_mask, dtype=np.uint8)

    # # Place markers at the coordinates of the local maxima
    # for x, y in local_maxima:
    #     markers[x, y] = 1

    # # Apply watershed
    # # TODO: check these parameters
    # watershed_spines = watershed(
    #     -distance_map,
    #     markers,
    #     mask=spine_mask,
    #     connectivity=np.ones((3, 3), dtype=bool),  # cross (+) neighbors
    #     compactness=0.01)

    # TODO: implement dilation of single-pixel markers?
    # spine_labels_test = morphology.isotropic_dilation(
    #     markers, radius=1)

    # Re-label components
    spine_labels = label(watershed_spines)
    dendrite_labels = label(dendrite_mask)

    # remove 0 label because it is background
    # unique_spine_labels = np.unique(spine_labels)[1:]
    # unique_dendrite_labels = np.unique(dendrite_labels)[1:]
    unique_spine_labels = tf.unique(
        tf.reshape(spine_labels, [-1]))[0][1:]
    unique_dendrite_labels = tf.unique(
        tf.reshape(dendrite_labels, [-1]))[0][1:]

    # spine_single_labels = np.array([None] * len(unique_spine_labels))
    # dendrite_single_labels = np.array([None] * len(unique_dendrite_labels))
    # spine_single_labels = [None] * tf.shape(unique_spine_labels)[0]
    # dendrite_single_labels = [None] * tf.shape(unique_dendrite_labels)[0]
    spine_single_labels = [None] * int(tf.shape(unique_spine_labels)[0])
    dendrite_single_labels = [None] * int(tf.shape(unique_dendrite_labels)[0])

    # Re-derive centroids from the segmented regions post-watershed
    # centroids = np.array([None] * len(unique_spine_labels))
    centroids = [None] * int(tf.shape(unique_spine_labels)[0])

    # for n in unique_spine_labels:
    #     cy, cx = calculate_centroid(spine_labels == n)
    #     centroids[n-1] = [cx, cy]
    #     binary_mask = (spine_labels == n).astype(np.uint8)
    #     if np.any(binary_mask):
    #         spine_single_labels[n-1] = binary_mask

    for i in range(tf.shape(unique_spine_labels)[0]):
        n = unique_spine_labels[i]
        cy, cx = calculate_centroid(tf.equal(spine_labels, n))
        centroids[i] = [cx, cy]
        binary_mask = tf.cast(tf.equal(spine_labels, n), tf.uint8)
        if tf.reduce_any(tf.cast(binary_mask, tf.bool)):
            spine_single_labels[i] = binary_mask.numpy()

    # spines_dicts = [None] * len(spine_single_labels)
    spines_dicts = defaultdict(lambda: defaultdict(dict))
    key = (roi.scanfield_index, roi.roi_index)

    for spine_n, spine in enumerate(spine_single_labels):
        spines_dicts[key][spine_n].append(
            {
                'roi_uuid': roi.roi.roiuuid,
                'centroid_pix': np.asarray(centroids[spine_n]),
                'mask': spine
            })

    # for n in unique_dendrite_labels:
    #     binary_mask = (dendrite_labels == n).astype(np.uint8)
    #     if np.any(binary_mask):
    #         dendrite_single_labels[n-1] = binary_mask

    for i in range(tf.shape(unique_dendrite_labels)[0]):
        n = unique_dendrite_labels[i]
        binary_mask = tf.cast(tf.equal(dendrite_labels, n), tf.uint8)
        if tf.reduce_any(tf.cast(binary_mask, tf.bool)):
            dendrite_single_labels[i] = binary_mask

    # dendrites_dicts = [None] * len(dendrite_single_labels)
    dendrites_dicts = defaultdict(lambda: defaultdict(dict))

    def skeletonize_wrapper(dendrite):
        return morphology.skeletonize(
            dendrite.numpy().astype(bool)).astype(np.uint8)

    for dendrite_n, dendrite in enumerate(dendrite_single_labels):
        skeletonized = tf.py_function(
            func=skeletonize_wrapper,
            inp=[dendrite],
            Tout=tf.uint8)

        indices = tf.where(skeletonized)
        rows, cols = indices[:, 0], indices[:, 1]
        skel_coords = [[c.numpy(), r.numpy()] for r, c in zip(rows, cols)]

        dendrites_dicts[key][dendrite_n] = {
            'roi_uuid': roi.roi.roiuuid,
            'dendrite_uuid': generate_uuid(),
            'mask': dendrite,
            'skeleton': skel_coords
        }
    print('check4')

    # skel_coords_fov = np.empty_like(dendrites_dicts[key])
    # for n, dendrite in enumerate(dendrites_dicts[key]):
    #     skel_coords_fov[n] = transform(
    #         [coords for coords in dendrite['skeleton']],
    #         np.array(roi.affine),
    #         np.array(roi.pixeltoreftransform),
    #         roi.centerxy,
    #         roi.pixelresolutionxy)

    # dendrites_dicts[key] = [{**dendrite, 'skeleton_fov': node.tolist()}
    #                         for dendrite, node in zip(
    #                                 dendrites_dicts[key].values(),
    #                                 skel_coords_fov)]

    skel_coords_fov_list = []

    for n, dendrite in enumerate(dendrites_dicts[key].values()):
        skeleton = dendrite['skeleton']
        if isinstance(skeleton, list):  # Ensure skeleton is a list
            coords = [[c, r] for r, c in skeleton]  # Correctly handle the coordinates
            coords = tf.convert_to_tensor(coords, dtype=tf.float32)

            affine = tf.convert_to_tensor(np.array(roi.affine), dtype=tf.float32)
            pixeltoreftransform = tf.convert_to_tensor(np.array(roi.pixeltoreftransform), dtype=tf.float32)
            centerxy = tf.convert_to_tensor(roi.centerxy, dtype=tf.float32)
            pixelresolutionxy = tf.convert_to_tensor(roi.pixelresolutionxy, dtype=tf.float32)

            transformed_coords = tf_transform(coords, affine, pixeltoreftransform, centerxy, pixelresolutionxy)
            skel_coords_fov_list.append(transformed_coords)

    # Convert list to TensorArray
    skel_coords_fov = tf.ragged.constant([coords.numpy() for coords in skel_coords_fov_list])

    dendrites_dicts[key] = [{**dendrite, 'skeleton_fov':
                             coords.numpy().tolist()}
                            for dendrite, coords in zip(
                                    dendrites_dicts[key].values(),
                                    skel_coords_fov)]

    # Refine masks using SD map
    # Create the sd_masks
    sd_centroids, sd_masks = create_sd_masks(
        roi.roi, kernel_size=kernel_size, sd_factor=sd_factor)

    # Calculate overlap with prediction masks
    intersection, intersection_indices = create_intersection_matrix(
        sd_masks, [spine['mask'] for spine in spines_dicts[key]])

    print('check5')

    # Move the prediction mask to the centroid of the overlapping sd_mask
    # for pair in intersection_indices:
    #     sd_index, pred_index = pair
    #     shifted_mask = move_mask_to_sd_centroid(
    #         spines_dicts[key][pred_index]['mask'],
    #         spines_dicts[key][pred_index]['centroid_pix'],
    #         sd_centroids[sd_index],)

    #     # Replace the mask in the dictionary
    #     spines_dicts[key][pred_index]['mask'] = shifted_mask
    #     cy, cx = calculate_centroid(shifted_mask)
    #     spines_dicts[key][pred_index]['centroid_pix'] = (cx, cy)

    for pair in intersection_indices:
        sd_index, pred_index = pair
        shifted_mask = move_mask_to_sd_centroid(
            tf.convert_to_tensor(
                spines_dicts[key][pred_index]['mask'], dtype=tf.float32),
            spines_dicts[key][pred_index]['centroid_pix'],
            sd_centroids[sd_index])

        # Replace the mask in the dictionary
        spines_dicts[key][pred_index]['mask'] = shifted_mask
        cy, cx = calculate_centroid(shifted_mask)
        spines_dicts[key][pred_index]['centroid_pix'] = (cx, cy)

    # Add new spines for non-overlapping sd_masks
    existing_sd_indices = [i[0] for i in intersection_indices]
    new_spine_id = len(spines_dicts[key])

    # for n, sd_mask in enumerate(sd_masks):
    #     if n not in existing_sd_indices:

    #         # Apply erosion followed by dilation to the new mask
    #         eroded_sd_mask = binary_erosion(
    #                               sd_mask, structure=np.ones((3, 3)))

    #         if np.any(eroded_sd_mask):
    #             # Apply dilation to the eroded mask
    #             dilated_sd_mask = binary_dilation(
    #                 eroded_sd_mask, structure=np.ones((3, 3)))

    #             new_spine_dict = {
    #                 'roi_uuid': roi.roi.roiuuid,
    #                 'centroid_pix': sd_centroids[n],
    #                 'mask': dilated_sd_mask
    #             }
    #             spines_dicts[key][new_spine_id].append(new_spine_dict)
    #             new_spine_id += 1

    for i in range(tf.shape(sd_masks)[0]):
        if i not in existing_sd_indices:
            eroded_sd_mask = tf.nn.erosion2d(
                tf.expand_dims(tf.expand_dims(sd_masks[i], 0), -1),
                filters=tf.ones((3, 3, 1, 1), dtype=tf.float32),
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            eroded_sd_mask = tf.squeeze(eroded_sd_mask, [0, -1])

            if tf.reduce_any(eroded_sd_mask):
                dilated_sd_mask = tf.nn.dilation2d(
                    tf.expand_dims(tf.expand_dims(eroded_sd_mask, 0), -1),
                    filters=tf.ones((3, 3, 1, 1), dtype=tf.float32),
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )
                dilated_sd_mask = tf.squeeze(dilated_sd_mask, [0, -1])

                new_spine_dict = {
                    'roi_uuid': roi.roi.roiuuid,
                    'centroid_pix': sd_centroids[i],
                    'mask': dilated_sd_mask
                }
                spines_dicts[key][new_spine_id].append(new_spine_dict)
                new_spine_id += 1

    # Convert list of masks to single label masks
    # all_spine_labels = np.zeros_like(spine_labels)
    # for spine in spines_dicts[key]:
    #     all_spine_labels[spine['mask'] > 0] = spine['spine_id'] + 1
    all_spine_labels = tf.zeros_like(spine_labels, dtype=tf.int32)

    for spine in spines_dicts[key]:
        all_spine_labels = tf.where(
            spine['mask'] > 0,
            spine['spine_id'] + 1,
            all_spine_labels)

    # Remove distant spines again after SD masks
    all_spine_labels = remove_distant_spines(
        all_spine_labels,
        dendrite_mask,
        dendrite_dilation_iterations
    )

    # Update spines_dicts with cleaned labels
    cleaned_spines_dicts = defaultdict(lambda: defaultdict(dict))
    # unique_labels = np.unique(all_spine_labels)[1:]  # Exclude background (0)
    unique_labels = tf.unique(tf.reshape(all_spine_labels, [-1]))[0][1:]

    # for lbl in unique_labels:
    #     mask = (all_spine_labels == lbl)
    #     if np.any(mask):
    #         cy, cx = calculate_centroid(mask)
    #         cleaned_spines_dicts[key][lbl].append({
    #             'roi_uuid': roi.roi.roiuuid,
    #             'spine_uuid': generate_uuid(),
    #             'centroid_pix': (cx, cy),
    #             'mask': mask
    #         })

    for lbl in unique_labels:
        mask = tf.equal(all_spine_labels, lbl)
        if tf.reduce_any(mask):
            cy, cx = calculate_centroid(mask)
            cleaned_spines_dicts[key][lbl].append({
                'roi_uuid': roi.roi.roiuuid,
                'spine_uuid': generate_uuid(),
                'centroid_pix': (cx, cy),
                'mask': mask
            })

    # centroids_fov = transform(
    #     [spine['centroid_pix']
    #      for spine in cleaned_spines_dicts[key].values()],
    #     np.array(roi.affine),
    #     np.array(roi.pixeltoreftransform),
    #     roi.centerxy,
    #     roi.pixelresolutionxy)
    centroids_fov = tf_transform(
        tf.convert_to_tensor(
            [spine['centroid_pix']
             for spine in cleaned_spines_dicts[key].values()]),
        tf.convert_to_tensor(roi.affine, dtype=tf.float32),
        tf.convert_to_tensor(roi.pixeltoreftransform, dtype=tf.float32),
        tf.convert_to_tensor(roi.centerxy, dtype=tf.float32),
        tf.convert_to_tensor(roi.pixelresolutionxy, dtype=tf.float32)
    )

    # Add z coordinate
    # centroids_fov = np.append(
    #     centroids_fov,
    #     float(roi.roi.name.split(', ')[0].split(' = ')[1]))
    z_coord = tf.convert_to_tensor(
       float(roi.roi.name.split(', ')[0].split(' = ')[1]), dtype=tf.float32)
    centroids_fov = tf.concat(
        [centroids_fov, tf.expand_dims(z_coord, 0)], axis=0)

    # Update spines_dicts with centroids_fov
    # for spine_id, centroid in zip(
    #         cleaned_spines_dicts[key].keys(), centroids_fov):
    #     cleaned_spines_dicts[key][spine_id]['centroid_fov'] = (
    #       centroid.tolist())

    for spine_id, centroid in zip(
            cleaned_spines_dicts[key].keys(), centroids_fov):
        cleaned_spines_dicts[key][spine_id]['centroid_fov'] = (
            centroid.numpy().tolist())

    return spines_dicts, dendrites_dicts


def process_predictions(
        segmenters: Union[tf.Tensor, List[tf.Tensor]],
        spine_predictions: Union[tf.Tensor, List[tf.Tensor]],
        dendrite_predictions: Union[tf.Tensor, List[tf.Tensor]],
        spine_threshold: float,
        dendrite_threshold: float,
        min_spine_size: float,
        mask_size: int,
        min_distance: int,
        min_dendrite_size: float,
        dendrite_dilation_iterations: int,
        kernel_size: int,
        sd_factor: int,
) -> Union[Tuple[List[dict], List[dict]],
           Tuple[List[List[dict]], List[List[dict]]]]:
    """
    Perform semantic segmentation to identify spines and dendrites
    for single or batch predictions.

    This function processes the predictions from the neural network to identify
    spines and dendrites within a single ROI or a batch of ROIs.
    It applies various image processing techniques to refine the predictions
    and generate final segmentation results.

    Parameters
    ----------
    segmenters : Union[tf.Tensor, List[tf.Tensor]]
        A single RoiSegmenter instance or a list of RoiSegmenter instances
        representing the regions of interest (ROIs).
    spine_predictions : Union[tf.Tensor, List[tf.Tensor]]
        The raw spine predictions from the neural network.
        Can be a single prediction or a batch of predictions.
    dendrite_predictions : Union[tf.Tensor, List[tf.Tensor]]
        The raw dendrite predictions from the neural network.
        Can be a single prediction or a batch of predictions.
    spine_threshold : float
        The threshold value to identify spines.
    dendrite_threshold : float
        The threshold value to identify dendrites.
    min_spine_size : float
        The minimum size to keep spines.
    mask_size : int
        The size of the mask used for certain operations.
    min_distance : int
        The minimum distance between detected centroids.
    min_dendrite_size : float
        The minimum size to keep dendrites.
    dendrite_dilation_iterations : int
        The number of iterations for dilating dendrites.
    kernel_size : int
        The size of the structuring element used for morphological operations.
    sd_factor : int
        Multiplier of the standard deviation to use as a binary threshold.

    Returns
    -------
    Union[Tuple[List[dict], List[dict]],
          Tuple[List[List[dict]], List[List[dict]]]]
        If processing a single ROI, returns a tuple of lists containing
        dictionaries with spine and dendrite information.
        If processing a batch of ROIs, returns a tuple of lists of lists
        containing dictionaries with spine and dendrite information
        for each ROI.

    Notes
    -----
    The function first checks if the predictions are for a batch of ROIs or a
    single ROI. It then processes each prediction accordingly
    by calling `process_single_prediction` for each pair of
    spine and dendrite predictions. The resulting segmentation information
    is compiled into dictionaries which include details about
    the identified spines and dendrites.
    """

    if isinstance(spine_predictions, list) or spine_predictions.ndim == 4:
        batch_spines_dicts = []
        batch_dendrites_dicts = []

        for spine_pred, dendrite_pred, roi in zip(
                spine_predictions, dendrite_predictions, segmenters):

            spines_dicts, dendrites_dicts = process_single_prediction(
                roi=roi,
                spine_prediction=spine_pred,
                dendrite_prediction=dendrite_pred,
                spine_threshold=spine_threshold,
                dendrite_threshold=dendrite_threshold,
                min_spine_size=min_spine_size,
                mask_size=mask_size,
                min_distance=min_distance,
                min_dendrite_size=min_dendrite_size,
                dendrite_dilation_iterations=dendrite_dilation_iterations,
                kernel_size=kernel_size,
                sd_factor=sd_factor)
            batch_spines_dicts.append(spines_dicts)
            batch_dendrites_dicts.append(dendrites_dicts)

        return batch_spines_dicts, batch_dendrites_dicts

    else:
        return process_single_prediction(
            roi=segmenters,
            spine_prediction=spine_predictions,
            dendrite_prediction=dendrite_predictions,
            spine_threshold=spine_threshold,
            dendrite_threshold=dendrite_threshold,
            min_spine_size=min_spine_size,
            mask_size=mask_size,
            min_distance=min_distance,
            min_dendrite_size=min_dendrite_size,
            dendrite_dilation_iterations=dendrite_dilation_iterations,
            kernel_size=kernel_size,
            sd_factor=sd_factor)


def threshold_prediction(
        prediction: tf.Tensor,
        threshold: float
) -> tf.Tensor:
    """
    Apply a threshold to the prediction to generate a binary mask.

    Parameters
    ----------
    prediction : tf.Tensor
        The prediction image.
    threshold : float
        The threshold value.

    Returns
    -------
    tf.Tensor
        The binary mask after applying the threshold.
    """
    return tf.cast(prediction > threshold, tf.uint8) * 255


# def remove_distant_spines(
#         labels: np.ndarray,
#         reference_prediction: np.ndarray,
#         iterations: int
# ) -> np.ndarray:

#     dilation = binary_dilation(
#         reference_prediction,
#         iterations=iterations)

#     unique_labels = np.unique(labels)
#     new_labels = np.zeros_like(labels)

#     for lbl in unique_labels:
#         if lbl == 0:
#             continue

#         label_mask = (labels == lbl)

#         if np.any(dilation & label_mask):
#             new_labels[label_mask] = lbl

#     return new_labels


def remove_distant_spines(
        labels: tf.Tensor,
        reference_prediction: tf.Tensor,
        iterations: int
) -> tf.Tensor:
    """
    Remove spine labels that are too distant from the reference prediction.
    tf.nn.dilation2d expects a 4-dimensional input,
    structured as [batch_size, height, width, channels].

    Parameters
    ----------
    labels : tf.Tensor
        The labeled image tensor.
    reference_prediction : tf.Tensor
        The reference prediction tensor.
    iterations : int
        The number of dilation iterations.

    Returns
    -------
    tf.Tensor
        The modified labeled image tensor.
    """
    # Ensure reference_prediction is uint8
    reference_prediction = tf.cast(reference_prediction, tf.uint8)

    # Ensure reference_prediction has the channel dimension
    reference_prediction = tf.expand_dims(reference_prediction, axis=-1)

    # Perform dilation
    dilation = tf.nn.dilation2d(
        input=tf.expand_dims(reference_prediction, axis=0),
        filters=tf.ones((3, 3, 1), dtype=tf.uint8),
        strides=[1, 1, 1, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format="NHWC",
    )
    dilation = tf.squeeze(dilation, axis=[0, -1])
    dilation = tf.cast(dilation, tf.bool)

    unique_labels = tf.unique(tf.reshape(labels, [-1])).y
    new_labels = tf.zeros_like(labels, dtype=tf.int32)

    for lbl in unique_labels:
        if lbl == 0:
            continue

        label_mask = tf.equal(labels, lbl)
        label_mask = tf.cast(label_mask, tf.bool)

        tf.debugging.assert_shapes([(dilation, tf.TensorShape([82, 50])),
                                    (label_mask, tf.TensorShape([82, 50]))])

        if tf.reduce_any(tf.logical_and(dilation, label_mask)):
            new_labels = tf.where(label_mask, labels, new_labels)

    return new_labels


# def remove_small_labels(
#         labels: np.ndarray,
#         min_size: int
# ) -> np.ndarray:
#     """
#     Remove objects smaller than the specified minimum size from labeled images.

#     Parameters:
#     ----------
#     labels : np.ndarray
#         The labeled image array where each unique value
#         represents a different object.
#     min_size : int
#         The minimum size threshold for objects to be retained.

#     Returns:
#     -------
#     labels : np.ndarray
#         The modified labeled image with small objects removed.
#     """
#     # Calculate the size of each unique label
#     component_sizes = np.bincount(
#         labels.ravel(), minlength=np.max(labels) + 1)

#     # Create a mask for small objects, excluding the background (label 0)
#     too_small = component_sizes < min_size
#     too_small[0] = False

#     # Use the mask to set small objects to 0
#     labels[too_small[labels]] = 0

#     return labels


def remove_small_labels(
        labels: tf.Tensor,
        min_size: int
) -> tf.Tensor:
    """
    Remove objects smaller than the specified minimum size from labeled images.

    Parameters:
    ----------
    labels : tf.Tensor
        The labeled image tensor where each unique value
        represents a different object.
    min_size : int
        The minimum size threshold for objects to be retained.

    Returns:
    -------
    labels : tf.Tensor
        The modified labeled image with small objects removed.
    """
    # Flatten the labels to calculate component sizes
    flat_labels = tf.reshape(labels, [-1])

    # Calculate the size of each unique label
    component_sizes = tf.math.bincount(
        flat_labels, minlength=tf.reduce_max(labels) + 1)

    # Create a mask for small objects, excluding the background (label 0)
    too_small = component_sizes < min_size
    too_small = tf.tensor_scatter_nd_update(too_small, [[0]], [False])

    # Use the mask to set small objects to 0
    too_small_mask = tf.gather(too_small, flat_labels)
    labels = tf.where(tf.reshape(too_small_mask, tf.shape(labels)), 0, labels)

    return labels


def tf_distance_transform(binary_image: tf.Tensor) -> tf.Tensor:
    """
    Perform distance transform on a binary image using TensorFlow.

    Parameters
    ----------
    binary_image : tf.Tensor
        The binary image tensor to transform.

    Returns
    -------
    tf.Tensor
        The distance transform of the binary image.
    """
    # Ensure binary_image is uint8
    uint8_image = tf.cast(binary_image, tf.uint8)

    # Convert to numpy array
    numpy_image = uint8_image.numpy()  # Convert to numpy array

    # Invert the image to ensure the foreground is white (255)
    inverted_image = 255 - numpy_image

    # OpenCV expects a single-channel 8-bit image
    single_channel_image = np.squeeze(inverted_image)

    # Calculate the distance transform using OpenCV
    distance_transform = cv2.distanceTransform(
        single_channel_image, cv2.DIST_L2, 5)

    # Convert back to TensorFlow tensor
    distance_map = tf.convert_to_tensor(distance_transform, dtype=tf.float32)

    return distance_map


def is_corner_joint(window: tf.Tensor) -> tf.Tensor:
    """
    Check if a 2x2 window contains a corner joint pattern using TensorFlow.

    Parameters
    ----------
    window : tf.Tensor
        A 2x2 tensor window from the labeled mask.

    Returns
    -------
    tf.Tensor
        A tensor indicating whether the window contains a corner joint pattern.
    """
    pattern1 = tf.logical_and(
        tf.logical_and(tf.not_equal(window[0, 1], 0), tf.not_equal(window[1, 0], 0)),
        tf.logical_and(tf.equal(window[0, 0], 0), tf.equal(window[1, 1], 0))
    )
    
    pattern2 = tf.logical_and(
        tf.logical_and(tf.not_equal(window[0, 0], 0), tf.not_equal(window[1, 1], 0)),
        tf.logical_and(tf.equal(window[0, 1], 0), tf.equal(window[1, 0], 0))
    )
    
    return tf.logical_or(pattern1, pattern2)


def remove_corner_joints(mask: tf.Tensor) -> tf.Tensor:
    """
    Remove corner joints in a labeled mask by identifying and modifying
    specific 2x2 patterns where non-zero values represent labels.

    Parameters
    ----------
    mask : tf.Tensor
        Labeled mask with connected components.

    Returns
    -------
    tf.Tensor
        Modified labeled mask with corner joints removed.
    """
    # Copy the mask to avoid modifying the original
    modified_mask = tf.identity(mask)

    # Get the shape of the mask
    mask_shape = tf.shape(modified_mask)

    # Iterate over the mask using a 2x2 moving window
    for y in range(mask_shape[0] - 1):
        for x in range(mask_shape[1] - 1):
            window = modified_mask[y:y+2, x:x+2]

            if is_corner_joint(window):
                update_indices = [
                    [y, x], [y, x+1],
                    [y+1, x], [y+1, x+1]
                ]
                modified_mask = tf.tensor_scatter_nd_update(
                    modified_mask,
                    update_indices,
                    tf.zeros([4], dtype=modified_mask.dtype))

    return modified_mask



# def calculate_intersection(
#         mask1: np.ndarray,
#         mask2: np.ndarray
# ) -> int:
#     """
#     Calculate the number of overlapping pixels between two binary masks.

#     Parameters
#     ----------
#     mask1 : np.ndarray
#         First binary mask.
#     mask2 : np.ndarray
#         Second binary mask.

#     Returns
#     -------
#     int
#         Number of overlapping pixels between the two masks.
#     """
#     if mask1.shape != mask2.shape:
#         raise ValueError("The two masks must have the same shape")

#     # Calculate intersection
#     intersection = np.logical_and(mask1, mask2)

#     # Return the count of overlapping pixels
#     return np.sum(intersection)


def calculate_intersection(
        mask1: tf.Tensor,
        mask2: tf.Tensor
) -> tf.Tensor:
    """
    Calculate the intersection between two binary masks.

    Parameters
    ----------
    mask1 : tf.Tensor
        The first binary mask.
    mask2 : tf.Tensor
        The second binary mask.

    Returns
    -------
    tf.Tensor
        The number of overlapping pixels between the two masks.
    """
    return tf.reduce_sum(tf.cast(mask1 & mask2, tf.int32))


# def create_intersection_matrix(
#         sd_masks: list[np.ndarray],
#         prediction_masks: list[np.ndarray]
# ) -> tuple:
#     """
#     Create an intersection matrix between two groups of binary masks and return
#     the indices of pairs of masks with an intersection greater than 0.

#     Parameters
#     ----------
#     sd_masks : list of np.ndarray
#         List of binary masks from the first group.
#     prediction_masks : list of np.ndarray
#         List of binary masks from the second group.

#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - np.ndarray: A matrix where element (i, j) is
#             the number of overlapping
#             pixels between sd_masks[i] and prediction_masks[j].
#         - list of tuple: List of tuples where each tuple contains the indices
#           (i, j) of the pairs of masks that have an overlap greater than 0.
#     """
#     num_sd_masks = len(sd_masks)
#     num_prediction_masks = len(prediction_masks)

#     intersection_matrix = np.zeros(
#         (num_sd_masks, num_prediction_masks), dtype=int)
#     overlap_indices = []

#     for i, sd_mask in enumerate(sd_masks):
#         for j, prediction_mask in enumerate(prediction_masks):
#             intersection = calculate_intersection(sd_mask, prediction_mask)
#             intersection_matrix[i, j] = intersection
#             if intersection > 0:
#                 overlap_indices.append((i, j))

#     return intersection_matrix, overlap_indices


def create_intersection_matrix(
        sd_masks: List[tf.Tensor],
        prediction_masks: List[tf.Tensor]
) -> Tuple[tf.Tensor, List[Tuple[int, int]]]:
    """
    Create an intersection matrix between two groups of binary masks and return
    the indices of pairs of masks with an intersection greater than 0.

    Parameters
    ----------
    sd_masks : List[tf.Tensor]
        List of binary masks from the first group.
    prediction_masks : List[tf.Tensor]
        List of binary masks from the second group.

    Returns
    -------
    Tuple[tf.Tensor, List[Tuple[int, int]]]
        A tuple containing:
        - tf.Tensor: A matrix where element (i, j) is the number of overlapping
          pixels between sd_masks[i] and prediction_masks[j].
        - List[Tuple[int, int]]: List of tuples of indices
          (i, j) of the pairs of masks that have an overlap greater than 0.
    """
    num_sd_masks = len(sd_masks)
    num_prediction_masks = len(prediction_masks)

    intersection_matrix = tf.zeros(
        (num_sd_masks, num_prediction_masks), dtype=tf.int32)
    overlap_indices = []

    for i, sd_mask in enumerate(sd_masks):
        for j, prediction_mask in enumerate(prediction_masks):
            intersection = calculate_intersection(sd_mask, prediction_mask)
            intersection_matrix = intersection_matrix.numpy()
            intersection_matrix[i, j] = intersection.numpy()
            intersection_matrix = tf.convert_to_tensor(
                intersection_matrix, dtype=tf.int32)
            if intersection > 0:
                overlap_indices.append((i, j))

    return intersection_matrix, overlap_indices


# def create_sd_masks(
#         roi: object,
#         kernel_size: int,
#         sd_factor: float,
# ) -> Tuple[np.ndarray, List[np.ndarray]]:
#     """
#     Create standard deviation masks for the region of interest (ROI)
#     and compute centroids and binary masks for each connected component.

#     Parameters
#     ----------
#     roi : list
#         List containing ROI data.
#     kernel_size : int
#         Size of the structuring element used for morphological operations.
#     sd_factor: float
#         Multiplier of SD to use as binary threshold

#     Returns
#     -------
#     Tuple[np.ndarray, List[np.ndarray]]
#         A tuple containing:
#         - sd_centroids: An array of centroids.
#         - spine_single_labels: A list of binary masks.
#     """

#     # half_sd_window = int(int(roi.frame_rate * 1) / 2)
#     # stim_frame = int(roi.frame_rate * 1)
#     # indexes_around_stim = np.arange(stim_frame - half_sd_window,
#     #                                 stim_frame + half_sd_window)

#     # Create binary masks based on standard deviation threshold
#     # sd_masks = []
#     # for n in roi.sweep_list:
#     #     frames_around_stim = np.array(
#     #         [roi[n].ch1[i].frame for i in indexes_around_stim])
#     #     std_proj_around_stim = np.std(frames_around_stim, axis=0)
#     #     sd_mask = (std_proj_around_stim >
#     #                np.mean(std_proj_around_stim) * sd_factor)
#     #     sd_masks.append(sd_mask)

#     sd_masks = [(roi[n].ch1.std.frame >
#                  np.mean(roi[n].ch1.std.frame) * sd_factor)
#                 for n in roi.sweep_list]

#     # Combine masks to get the maximum mask
#     sd_mask_max = np.max(sd_masks, axis=0)

#     # Morphological operations
#     selem = morphology.square(kernel_size)
#     sd_mask_max = morphology.binary_dilation(sd_mask_max, selem)
#     sd_mask_max = morphology.binary_erosion(sd_mask_max, selem)

#     # Remove small objects
#     labeled_array, num_features = ndimage.label(sd_mask_max)
#     component_sizes = ndimage.sum(
#         sd_mask_max, labeled_array, range(num_features + 1))
#     size_mask = component_sizes > 1
#     size_mask[0] = 0  # Keep background labeled as 0

#     cleaned_eroded = size_mask[labeled_array]
#     sd_mask_max = cleaned_eroded.astype(bool)

#     # Label connected components
#     sd_mask_max_labeled = label(sd_mask_max)
#     unique_spine_labels = np.unique(sd_mask_max_labeled)[1:]

#     # Initialize arrays for centroids and binary masks
#     centroids = []
#     spine_single_labels = []

#     for n in unique_spine_labels:
#         binary_mask = (sd_mask_max_labeled == n).astype(np.uint8)
#         if np.any(binary_mask):
#             cy, cx = calculate_centroid(binary_mask)
#             centroids.append([cx, cy])
#             spine_single_labels.append(binary_mask)

#     return np.array(centroids), spine_single_labels


def create_sd_masks(
        roi: object,
        kernel_size: int,
        sd_factor: float
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Create standard deviation masks for the region of interest (ROI)
    and compute centroids and binary masks for each connected component.

    Parameters
    ----------
    roi : list
        List containing ROI data.
    kernel_size : int
        Size of the structuring element used for morphological operations.
    sd_factor: float
        Multiplier of SD to use as binary threshold.

    Returns
    -------
    Tuple[tf.Tensor, List[tf.Tensor]]
        A tuple containing:
        - sd_centroids: A tensor of centroids.
        - spine_single_labels: A list of binary masks.
    """

    # Create binary masks based on standard deviation threshold
    sd_masks = [(tf.cast(roi[n].ch1.std.frame, tf.float32) >
                 tf.reduce_mean(
                     tf.cast(roi[n].ch1.std.frame, tf.float32)) * sd_factor)
                for n in roi.sweep_list]

    # Convert boolean masks to float32
    sd_masks = [tf.cast(mask, tf.float32) for mask in sd_masks]

    # Combine masks to get the maximum mask
    sd_mask_max = tf.reduce_max(tf.stack(sd_masks), axis=0)

    # Morphological operations using TensorFlow
    filter_shape = (kernel_size, kernel_size, 1)  # Correct filter shape

    sd_mask_max = tf.nn.dilation2d(
        tf.expand_dims(tf.expand_dims(sd_mask_max, 0), -1),
        tf.ones(filter_shape, dtype=tf.float32),
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format="NHWC",
        dilations=[1, 1, 1, 1]
    )
    sd_mask_max = tf.nn.erosion2d(
        sd_mask_max,
        tf.ones(filter_shape, dtype=tf.float32),
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format="NHWC",
        dilations=[1, 1, 1, 1]
    )
    sd_mask_max = tf.squeeze(sd_mask_max, [0, -1])

    # Convert the mask to binary image
    binary_mask = tf.cast(tf.greater(sd_mask_max, 0), tf.uint8)

    # Label connected components
    sd_mask_max_labeled = label(binary_mask)

    # Remove small objects
    num_features = tf.reduce_max(sd_mask_max_labeled)
    component_sizes = tf.math.bincount(tf.reshape(sd_mask_max_labeled, [-1]), minlength=num_features + 1)
    size_mask = component_sizes > 1
    size_mask = tf.tensor_scatter_nd_update(size_mask, [[0]], [False])

    cleaned_eroded = tf.gather(size_mask, tf.reshape(sd_mask_max_labeled, [-1]))
    sd_mask_max = tf.reshape(cleaned_eroded, tf.shape(sd_mask_max))

    # Label connected components again after removing small objects
    sd_mask_max_labeled = label(tf.cast(sd_mask_max > 0, tf.uint8))

    unique_spine_labels = tf.unique(tf.reshape(sd_mask_max_labeled, [-1]))[0]
    unique_spine_labels = unique_spine_labels[1:]  # Exclude background

    # Initialize arrays for centroids and binary masks
    centroids = []
    spine_single_labels = []

    for n in unique_spine_labels:
        binary_mask = tf.cast(tf.equal(sd_mask_max_labeled, n), tf.uint8)
        if tf.reduce_any(binary_mask):
            cy, cx = calculate_centroid(binary_mask)
            centroids.append([cx, cy])
            spine_single_labels.append(binary_mask)

    return (
        tf.convert_to_tensor(centroids, dtype=tf.float32),
        spine_single_labels
    )


# def move_mask_to_sd_centroid(
#         mask: np.ndarray,
#         current_centroid: Tuple[int, int],
#         new_centroid: Tuple[int, int]
# ) -> np.ndarray:
#     """
#     Move a binary mask to a new centroid position.

#     Parameters
#     ----------
#     mask : np.ndarray
#         Binary mask with a single component.
#     current_centroid : Tuple[int, int]
#         Current centroid (y, x) of the mask.
#     new_centroid : Tuple[int, int]
#         New centroid (y, x) to which the mask should be moved.

#     Returns
#     -------
#     np.ndarray
#         Binary mask moved to the new centroid.
#     """
#     # Calculate the shift required
#     shift_y = new_centroid[0] - current_centroid[0]
#     shift_x = new_centroid[1] - current_centroid[1]

#     # Apply the shift to the mask
#     shifted_mask = ndimage.shift(
#         mask.astype(float),
#         shift=(shift_y, shift_x),
#         order=0,
#         mode='constant',
#         cval=0)

#     return shifted_mask > 0


def move_mask_to_sd_centroid(
        mask: tf.Tensor,
        current_centroid: Tuple[int, int],
        new_centroid: Tuple[int, int]
) -> tf.Tensor:
    """
    Move a binary mask to a new centroid position.

    Parameters
    ----------
    mask : tf.Tensor
        Binary mask with a single component.
    current_centroid : Tuple[int, int]
        Current centroid (y, x) of the mask.
    new_centroid : Tuple[int, int]
        New centroid (y, x) to which the mask should be moved.

    Returns
    -------
    tf.Tensor
        Binary mask moved to the new centroid.
    """
    # Calculate the shift required
    shift_y = new_centroid[0] - current_centroid[0]
    shift_x = new_centroid[1] - current_centroid[1]

    # Create a translation matrix
    translation_matrix = tf.convert_to_tensor([[1, 0, shift_x],
                                               [0, 1, shift_y],
                                               [0, 0, 1]], dtype=tf.float32)

    # Apply the shift to the mask
    mask_shape = tf.shape(mask)
    mask_indices = tf.where(mask > 0)
    ones = tf.ones((tf.shape(mask_indices)[0], 1), dtype=tf.float32)
    mask_indices_homogeneous = tf.concat(
        [tf.cast(mask_indices, tf.float32), ones], axis=1)
    shifted_indices_homogeneous = tf.matmul(
        mask_indices_homogeneous, translation_matrix, transpose_b=True)
    shifted_indices = tf.cast(shifted_indices_homogeneous[:, :2], tf.int32)

    # Create a new mask with the shifted indices
    shifted_mask = tf.scatter_nd(
        shifted_indices, tf.ones(tf.shape(shifted_indices)[0]), mask_shape)

    return shifted_mask


# def calculate_centroid(
#         mask: np.ndarray
# ) -> Tuple[int, int]:

#     M = moments(mask)
#     cy, cx = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]

#     return (cy, cx)


def calculate_centroid(mask: tf.Tensor) -> Tuple[int, int]:
    """
    Calculate the centroid of a binary mask using TensorFlow.

    Parameters
    ----------
    mask : tf.Tensor
        The binary mask.

    Returns
    -------
    Tuple[int, int]
        The (cy, cx) coordinates of the centroid.
    """
    # Ensure mask is float32 for calculations
    mask = tf.cast(mask, tf.float32)

    # Get the coordinates of the mask
    coords = tf.where(mask > 0)

    # Calculate moments
    total_mass = tf.reduce_sum(mask)
    cy = tf.reduce_sum(coords[:, 0] * tf.gather_nd(mask, coords)) / total_mass
    cx = tf.reduce_sum(coords[:, 1] * tf.gather_nd(mask, coords)) / total_mass

    return (cy.numpy(), cx.numpy())


def create_spines_dict(
        labels: np.ndarray,
        unique_labels: np.ndarray,
) -> List[dict]:
    """Create a list of dictionaries with spine information."""

    spines_dicts = []

    for lbl in unique_labels:
        binary_mask = (labels == lbl).astype(np.uint8)
        if np.any(binary_mask):
            cy, cx = calculate_centroid(binary_mask)
            spine_dict = {
                'spine_id': lbl - 1,
                'centroid_pix': (cx, cy),
                'mask': binary_mask
            }
            spines_dicts.append(spine_dict)

    return spines_dicts


def create_dendrites_dict(
        labels: np.ndarray,
        unique_labels: np.ndarray
) -> List[dict]:
    """Create a list of dictionaries with segment information."""

    dendrites_dicts = []

    for lbl in unique_labels:
        binary_mask = (labels == lbl).astype(np.uint8)
        if np.any(binary_mask):
            dendrite_dict = {
                'ROI_id': lbl - 1,
                'mask': binary_mask
            }
            dendrites_dicts.append(dendrite_dict)

    return dendrites_dicts
