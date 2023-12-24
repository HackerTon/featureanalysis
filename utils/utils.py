import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataloader.dataloader import UAVIDDataset


def combine_channels(image: torch.Tensor, colors: np.ndarray, is_predict: bool):
    _, h, w = image.shape
    output_image = torch.zeros([h, w, 3], dtype=torch.uint8)
    for i in range(colors.shape[0]):
        if is_predict:
            mask = image[i] > 0.5
        else:
            mask = image[i] == 1
        output_image[mask] = colors[i]
    return output_image


def visualize(
    input_image: torch.Tensor,
    grouth_truth: torch.Tensor,
    predicted: torch.Tensor,
):
    colors = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [128, 64, 128],
            [0, 128, 0],
            [128, 128, 0],
            [64, 0, 128],
            [192, 0, 192],
            [0, 0, 128],
        ],
        dtype=np.uint8,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=200)
    legend_patches = [
        patches.Patch(
            color=np.concatenate([color / 255, [1]]),
            label=UAVIDDataset.dataset_labels[idx],
        )
        for idx, color in enumerate(colors)
    ]
    fig.legend(handles=legend_patches, bbox_to_anchor=(1, 0.5))
    grouth_truth_image = combine_channels(grouth_truth, colors, False)
    predicted_image = combine_channels(predicted, colors, True)
    input_image = torch.permute(input_image[0], [1, 2, 0])

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()

    axes[0].set_title("Input Image")
    axes[1].set_title("Grouth Truth Image")
    axes[2].set_title("Predicted Image")

    axes[0].imshow(input_image)
    axes[1].imshow(grouth_truth_image)
    axes[2].imshow(predicted_image)
