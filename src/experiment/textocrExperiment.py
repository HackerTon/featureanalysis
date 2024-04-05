from typing import Tuple

import torch
from src.experiment.experimentbase import ExperimentBase
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import crop, resize

from src.dataloader.dataloader import TextOCRDataset
from src.dataloader.transform import ToNormalized
from src.model.model import BackboneType, MultiNet
from src.service.hyperparamater import Hyperparameter


class TextocrExperiment(ExperimentBase):
    def __init__(self, hyperparameter: Hyperparameter, device: str) -> None:
        super().__init__()

        raise Exception('Textocr experiment not implemented')

        # Initialization
        self.train_dataloader, self.test_dataloader = create_textocr_dataloader(
            path=hyperparameter.data_path,
            batch_size=hyperparameter.batch_size_train,
        )
        self.model = MultiNet(numberClass=2, backboneType=BackboneType.RESNET50)
        self.preprocessor = v2.Compose(
            [
                ToNormalized(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Move weights to specified device
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=hyperparameter.learning_rate,
            fused=True if device == "cuda" else False,
        )


random_generator = torch.Generator().manual_seed(1234)


def train_collate_fn(data):
    if torch.rand(1, generator=random_generator)[0] > 0.5:
        current_size = 512
    else:
        current_size = 256
    images = []
    labels = []

    # If current_size is the same size as input
    # skip cropping
    if (data[0][0].size(1) == current_size) and (data[0][0].size(2) == current_size):
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


def test_collate_fn(data):
    images = []
    labels = []
    for x in data:
        image, label = x
        image = image
        label = label
        images.append(resize(image, [512, 512]))
        labels.append(resize(label, [512, 512]))
    return (torch.stack(images), torch.stack(labels))


def create_textocr_dataloader(
    path: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextOCRDataset(path, True)
    # train_dataset = Subset(train_dataset, [x for x in range(1)])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=train_collate_fn,
    )
    test_dataset = TextOCRDataset(path, False)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=test_collate_fn,
    )
    return train_dataloader, test_dataloader
