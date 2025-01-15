""" Created on Fri Mar 22 13:14:34 2024
    @author: dcupolillo """

from pathlib import Path
import matplotlib.pyplot as plt
import imageio as io
import numpy as np

folder_path = Path('Y:\\SynEmo\\prova')

raters = ['dario', 'federica', 'vincenzo']
spines = {rater: [] for rater in raters}
dendrites = {rater: [] for rater in raters}

for rater in raters:
    rater_folder_path = folder_path / rater
    if rater_folder_path.is_dir():
        for file in rater_folder_path.iterdir():
            if file.suffix in ['.tif', '.tiff']:
                if 'dendrite' in file.stem:
                    dendrites[rater].append(file)
                elif 'spines' in file.stem:
                    spines[rater].append(file)

index = 5
files = [files[index] for rater, files in spines.items()]

images = np.array([io.mimread(file) for file in files])
fig, axes = plt.subplots(1, len(raters) + 2, sharex=True, sharey=True)
for n, (ax, file, rater, image) in enumerate(
        zip(axes[:-2], files, raters, images)):
    ax.imshow(image[0], cmap='binary_r')
    ax.set_title(rater)
    images[n] = image

consensus_image = np.logical_and.reduce(images)  # Pixel-wise logical AND
axes[-2].imshow(consensus_image[0], cmap='binary_r')
axes[-2].set_title('Consensus')

agreement_level = np.sum(images, axis=0)
agreement_map = agreement_level / len(files)
axes[-1].imshow(agreement_map[0], cmap='viridis')
# axes[-1].colorbar(label='Agreement Level')
axes[-1].set_title('Pixel-Wise\nAgreement Map')

plt.tight_layout()