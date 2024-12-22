from typing import Tuple

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import crop, five_crop

# from src.dataloader.dataset.he_dataset import HeDataset
from src.dataloader.dataset.he_dataset_direct import HeDataset
from src.dataloader.transform import ImagenetNormalize, ToNormalized
from src.experiment.experimentbase import ExperimentBase
from src.model.model import UNETNetwork
from src.service.parameter import Parameter


class HeExperiment(ExperimentBase):
    def __init__(self, parameter: Parameter) -> None:
        super().__init__()

        self.train_dataloader, self.test_dataloader = create_dataloader(
            path=parameter.data_path,
            batch_size=parameter.batch_size_train,
        )

        self.model = UNETNetwork(numberClass=3)
        self.preprocessor = v2.Compose(
            [
                ToNormalized(),
                ImagenetNormalize(),
            ]
        )

        # Move weights to specified device
        self.model = self.model.to(parameter.device)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=parameter.learning_rate,
            fused=True if parameter.device == "cuda" else False,
        )
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=self.optimizer,
        #     max_lr=hyperparameter.learning_rate,
        #     steps_per_epoch=len(self.train_dataloader),
        #     epochs=hyperparameter.epoch,
        # )


def cross_entropy_dirichlet(prediction: torch.Tensor, target: torch.Tensor):
    alpha = prediction + 1
    diriclet_strength = alpha.sum(dim=1, keepdim=True)
    return (target * (torch.digamma(diriclet_strength) - torch.digamma(alpha))).sum(1)


def KL_divergence_dirichlet(prediction: torch.Tensor, target: torch.Tensor):
    alpha = prediction + 1
    n_class = torch.tensor(prediction.size(1))
    approx_alpha = target + (1 - target) * alpha

    first_term = torch.lgamma(approx_alpha.sum(dim=1))
    first_term -= torch.lgamma(n_class) + torch.lgamma(approx_alpha).sum(dim=1)
    second_term = (
        (approx_alpha - 1)
        * (
            torch.digamma(approx_alpha)
            - torch.digamma(approx_alpha.sum(dim=1, keepdim=True))
        )
    ).sum(dim=1)
    return first_term + second_term


def overall_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_t: torch.Tensor,
):
    prediction = prediction.relu()
    loss = cross_entropy_dirichlet(prediction, target)
    loss += lambda_t * KL_divergence_dirichlet(
        prediction,
        target,
    )
    return loss.mean()


random_generator = torch.Generator().manual_seed(1234)


def train_collate(data):
    current_size = 512
    images = []
    labels = []

    # If current_size is the same size as input
    # skip cropping
    if data[0][0].size(1) == current_size:
        for x in data:
            image, label = x
            images.append(image)
            labels.append(label)
    else:
        for x in data:
            image, label = x
            i, j, h, w = v2.RandomCrop.get_params(image, (current_size, current_size))
            images.append(crop(image, i, j, h, w))
            labels.append(crop(label, i, j, h, w))
    return (torch.stack(images), torch.stack(labels))


def test_collate(data):
    images = []
    labels = []
    for x in data:
        image, label = x
        split_image = five_crop(image, [512, 512])
        split_label = five_crop(label, [512, 512])
        for i in range(5):
            images.append(split_image[i])
            labels.append(split_label[i])
    return (torch.stack(images), torch.stack(labels))


def create_dataloader(
    path: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    global_dataset = HeDataset(directory_path=path)
    SPLIT_PERCENTAGE = 0.8

    train_dataset, test_dataset = random_split(
        global_dataset,
        [SPLIT_PERCENTAGE, 1 - SPLIT_PERCENTAGE],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=train_collate,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        # collate_fn=test_collate,
        batch_size=8,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader
