""" Created on Tue Jun 11 10:16:41 2024
    @author: dcupolillo """

import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

def load_masks(mask_paths: List[Path]) -> List[np.ndarray]:
    """
    Load binary masks from the provided file paths.

    Parameters
    ----------
    mask_paths : List[Path]
        List of paths to the mask files.

    Returns
    -------
    List[np.ndarray]
        List of binary mask arrays.
    """
    masks = []
    for path in mask_paths:
        if path.suffix == '.npy':
            mask = np.load(path)
        elif path.suffix in ['.tif', '.tiff']:
            mask = np.array(Image.open(path))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        masks.append(mask)
    return masks


def calculate_iou(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        The predicted binary mask.
    gt_mask : np.ndarray
        The ground truth binary mask.

    Returns
    -------
    float
        The IoU score.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def evaluate_model(
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
) -> float:
    """
    Evaluate the model performance by calculating the
    average IoU across all images.

    Parameters
    ----------
    pred_masks : List[np.ndarray]
        List of predicted binary masks.
    gt_masks : List[np.ndarray]
        List of ground truth binary masks.

    Returns
    -------
    float
        The average IoU score.
    """
    iou_scores = []
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        iou = calculate_iou(pred_mask, gt_mask)
        iou_scores.append(iou)
    return np.mean(iou_scores)


def create_iou_heatmap(
        pred_masks: List[np.ndarray],
        gt_masks: List[List[np.ndarray]],
        rater_names: List[str]
) -> None:
    """
    Create a heatmap matrix of IoU scores between predicted masks and each of the human raters.

    Parameters
    ----------
    pred_masks : List[np.ndarray]
        List of predicted binary masks.
    gt_masks : List[List[np.ndarray]]
        List of lists of ground truth binary masks for each rater.
    rater_names : List[str]
        List of rater names.
    """
    num_images = len(pred_masks)
    num_raters = len(gt_masks)
    
    iou_matrix = np.zeros((num_raters, num_images))

    for i in range(num_raters):
        for j in range(num_images):
            iou_matrix[i, j] = calculate_iou(pred_masks[j], gt_masks[i][j])
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(iou_matrix, annot=True, xticklabels=[f'Image {i+1}' for i in range(num_images)], yticklabels=rater_names, cmap='viridis')
    plt.title('IoU Heatmap: Predicted Masks vs Human Raters')
    plt.xlabel('Images')
    plt.ylabel('Human Raters')
    plt.show()

    return iou_matrix



# %%

# Example usage:
pred_mask_paths = [Path(r'neuralnetwork\IoU\240130_0_1.npy'),
                   Path(r'neuralnetwork\IoU\240201_4_3.npy'),
                   Path(r'neuralnetwork\IoU\240130_17_2.npy')]

gt_mask_paths_V = [Path(r'neuralnetwork\IoU\240130_0_1_V.tif'),
                   Path(r'neuralnetwork\IoU\240201_4_3_V.tif'),
                   Path(r'neuralnetwork\IoU\240130_17_2_V.tif')]
gt_mask_paths_S = [Path(r'neuralnetwork\IoU\240130_0_1_S.tif'),
                   Path(r'neuralnetwork\IoU\240201_4_3_S.tif'),
                   Path(r'neuralnetwork\IoU\240130_17_2_S.tif')]
gt_mask_paths_A = [Path(r'neuralnetwork\IoU\240130_0_1_A.tif'),
                   Path(r'neuralnetwork\IoU\240201_4_3_A.tif'),
                   Path(r'neuralnetwork\IoU\240130_17_2_A.tif')]

pred_masks = load_masks(pred_mask_paths)
gt_masks_V = load_masks(gt_mask_paths_V)
gt_masks_S = load_masks(gt_mask_paths_S)
gt_masks_A = load_masks(gt_mask_paths_A)

# %%

gt_masks_all = [gt_masks_V, gt_masks_S, gt_masks_A]
rater_names = ['Rater V', 'Rater S', 'Rater A']

iou_matrix = create_iou_heatmap(pred_masks, gt_masks_all, rater_names)

all_masks = gt_masks_V + gt_masks_S + gt_masks_A + pred_masks
all_names = ['V1', 'V2', 'V3', 'S1', 'S2', 'S3', 'A1', 'A2', 'A3', 'Model1', 'Model2', 'Model3']
intra_rater_iou_matrix = create_intra_rater_iou_matrix(all_masks, all_names)

V, S, A = iou_matrix

colors = ['red', 'green', '#0072AF']

fig, ax = plt.subplots(figsize=(12, 4))
for data, label, color in zip(iou_matrix, labels, colors):
    ax.plot(np.arange(3),
            data,
            lw=2,
            clip_on=False,
            label=label,
            color=color,
)
ax.plot(np.arange(3),
        np.mean(iou_matrix, axis = 0),
        lw=4,
        marker='o',
        markersize=10,
        color='black',
        clip_on=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, 0.5)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 20))
ax.spines[['bottom', 'left']].set_linewidth(2)

ax.xaxis.set_tick_params(width=2, length=10)
ax.yaxis.set_tick_params(width=2, length=10)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['V', 'S', 'A'])
ax.set_xlabel('Raters', labelpad=15)
ax.set_ylabel('IoU', labelpad=15)
plt.tight_layout()

plt.savefig(
    Path(r'Y:\SynEmo\figures4poster\IoU_plot.png'),
    dpi=1000)

# %%

gt_masks_all = [gt_masks_V, gt_masks_S, gt_masks_A]
rater_names = ['V', 'S', 'A', 'Mod.']

# Create IoU matrix for heatmap
num_images = len(pred_masks)
num_raters = len(gt_masks_all) + 1  # Including model

iou_matrix = np.zeros((num_raters, num_raters))

# Fill the IoU matrix
for i, gt_masks in enumerate(gt_masks_all):
    for j in range(num_images):
        iou_matrix[i, -1] += calculate_iou(gt_masks[j], pred_masks[j])
    iou_matrix[i, -1] /= num_images
    iou_matrix[-1, i] = iou_matrix[i, -1]  # Symmetric entry

for i in range(num_raters - 1):
    iou_matrix[i, i] = 1.0  # Diagonal for intra-rater comparison should be 1
    for j in range(i + 1, num_raters - 1):
        for k in range(num_images):
            iou_matrix[i, j] += calculate_iou(gt_masks_all[i][k], gt_masks_all[j][k])
        iou_matrix[i, j] /= num_images
        iou_matrix[j, i] = iou_matrix[i, j]  # Symmetric entry

iou_matrix[-1, -1] = 1.0  # Model's diagonal should be 1

# Plot the heatmap
plt.figure(figsize=(10, 7))
fontsize=50
heatmap = plt.imshow(iou_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
colorbar = plt.colorbar(heatmap)
colorbar.ax.tick_params(labelsize=fontsize)
colorbar.set_label('IoU', fontsize=fontsize, labelpad=15)
plt.xticks(ticks=np.arange(num_raters), labels=rater_names, fontsize=fontsize, rotation=45)
plt.yticks(ticks=np.arange(num_raters), labels=rater_names, fontsize=fontsize, rotation=45, va='center')
plt.tight_layout()

# plt.savefig(
#     Path(r'Y:\SynEmo\figures4poster\IoU_heatmap.png'),
#     dpi=1000)

# %%