import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import InterpolationMode, crop, resize
from torchvision.utils import draw_segmentation_masks

from dataloader.dataloader import CardiacDatasetHDF5, TextOCRDataset
from dataloader.transform import ToNormalized
from loss import dice_index, total_loss
from model.model import BackboneType, MultiNet
from service.hyperparamater import Hyperparameter
from service.model_saver_service import ModelSaverService


class Trainer:
    def __init__(self, train_report_rate: float = 0.1) -> None:
        """
        train_report_rate: float = [0.0, 1.0]
        """
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.writer_train = SummaryWriter(
            "data/log/training/train_{}".format(timestamp)
        )
        self.writer_test = SummaryWriter("data/log/training/test_{}".format(timestamp))
        self.model_saver = ModelSaverService(path=Path("data/model"), topk=2)
        self.train_report_rate = train_report_rate

    def run_trainer(
        self,
        device: str,
        hyperparameter: Hyperparameter,
        experiment_num: int,
    ):
        if experiment_num == 0:
            # Initialization
            train_dataloader, test_dataloader = create_textocr_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            model = MultiNet(numberClass=2, backboneType=BackboneType.RESNET50)
            preprocessor = v2.Compose(
                [
                    ToNormalized(),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            # Move weights to specified device
            model = model.to(device)
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=hyperparameter.learning_rate,
                fused=True if device == "cuda" else False,
            )

            # Run
            self.train(
                epochs=hyperparameter.epoch,
                model=model,
                dataloader_train=train_dataloader,
                dataloader_test=test_dataloader,
                optimizer=optimizer,
                loss_fn=total_loss,
                preprocess=preprocessor,
                device=device,
            )
        elif experiment_num == 5:
            # Initialization
            train_dataloader, test_dataloader = create_cardiac_dataloader_traintest(
                path=hyperparameter.data_path,
                path2=hyperparameter.data_path2,
                batch_size=hyperparameter.batch_size_train,
            )
            model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
            preprocessor = v2.Compose(
                [
                    ToNormalized(),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            # Move weights to specified device
            model = model.to(device)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=hyperparameter.learning_rate,
                fused=True if device == "cuda" else False,
            )

            # Run
            self.train(
                epochs=hyperparameter.epoch,
                model=model,
                dataloader_train=train_dataloader,
                dataloader_test=test_dataloader,
                optimizer=optimizer,
                loss_fn=total_loss,
                preprocess=preprocessor,
                device=device,
            )
        else:
            print(f"Your experiment number ({experiment_num}) not found")

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess: v2.Compose,
        device: str,
    ):
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}, ", end="")

            # Unfreeze backbone at epoch 2
            if epoch == 2:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad = True

            initial_time = time.time()
            self._train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                preprocess=preprocess,
                device=device,
                dtype=dtype,
            )
            time_taken = time.time() - initial_time
            print(f"time_taken: {time_taken}s")

            if dataloader_test is not None:
                self._eval_one_epoch(
                    epoch=epoch,
                    model=model,
                    dataloader=dataloader_test,
                    loss_fn=loss_fn,
                    preprocess=preprocess,
                    device=device,
                    train_dataset_length=len(dataloader_train),
                    dtype=dtype,
                )
                self._visualize_one_epoch(
                    epoch=epoch,
                    model=model,
                    dataloader=dataloader_test,
                    preprocess=preprocess,
                    train_dataset_length=len(dataloader_train),
                    device=device,
                )
            self._save(model=model, epoch=epoch)

    def _save(self, model: torch.nn.Module, epoch: int):
        self.model_saver.save(model, epoch)

    def _train_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess: v2.Compose,
        device: str,
        dtype,
    ):
        rate_to_print = math.floor(len(dataloader) * self.train_report_rate)
        running_loss = 0.0
        running_iou = 0.0

        if device != "mps":
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
            for index, data in enumerate(dataloader):
                with torch.autocast(
                    device_type=device, dtype=dtype, enabled=device != "mps"
                ):
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = preprocess(inputs, labels)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    iou_score = dice_index(outputs.sigmoid(), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
                running_iou += iou_score.item()

                if index % rate_to_print == (rate_to_print - 1):
                    current_training_sample = epoch * len(dataloader) + index + 1
                    self.writer_train.add_scalar(
                        "loss",
                        running_loss / rate_to_print,
                        current_training_sample,
                    )
                    self.writer_train.add_scalar(
                        "iou_score",
                        running_iou / rate_to_print,
                        current_training_sample,
                    )
                    running_loss = 0.0
                    running_iou = 0.0
        else:
            for index, data in enumerate(dataloader):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = preprocess(inputs, labels)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                iou_score = dice_index(outputs.sigmoid(), labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
                running_iou += iou_score.item()

                if index % rate_to_print == (rate_to_print - 1):
                    current_training_sample = epoch * len(dataloader) + index + 1
                    self.writer_train.add_scalar(
                        "loss",
                        running_loss / rate_to_print,
                        current_training_sample,
                    )
                    self.writer_train.add_scalar(
                        "iou_score",
                        running_iou / rate_to_print,
                        current_training_sample,
                    )
                    running_loss = 0.0
                    running_iou = 0.0

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        preprocess: v2.Compose,
        loss_fn,
        device: str,
        train_dataset_length: int,
        dtype,
    ):
        sum_loss = 0.0
        sum_iou = 0.0

        if device != "mps":
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=dtype):
                    for data in dataloader:
                        inputs: torch.Tensor
                        labels: torch.Tensor
                        inputs, labels = data

                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs, labels = preprocess(inputs, labels)

                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        iou_score = dice_index(outputs.sigmoid(), labels)

                        sum_loss += loss.item()
                        sum_iou += iou_score.item()
        else:
            with torch.no_grad():
                for data in dataloader:
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = preprocess(inputs, labels)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    iou_score = dice_index(outputs.sigmoid(), labels)

                    sum_loss += loss.item()
                    sum_iou += iou_score.item()

        iteration = (epoch + 1) * train_dataset_length
        avg_loss = sum_loss / len(dataloader)
        avg_iou = sum_iou / len(dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)
        self.writer_test.add_scalar("iou_score", avg_iou, iteration)

    def _visualize_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: Union[torch.device, str],
        preprocess: v2.Compose,
        train_dataset_length: int,
    ):
        with torch.no_grad():
            for data in dataloader:
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                original_image = inputs
                inputs, labels = preprocess(inputs, labels)

                outputs = model(inputs)
                colors = [
                    (0, 0, 128),
                    (128, 64, 128),
                    (0, 128, 0),
                    (0, 128, 128),
                    (128, 0, 64),
                    (192, 0, 192),
                    (128, 0, 0),
                ]

                visualization_image = original_image[0]
                for i in range(outputs.size(1) - 1):
                    # Visualization for label
                    visualization_image = draw_segmentation_masks(
                        visualization_image,
                        labels[0, i + 1] > 0.5,
                        colors=colors[i],
                        alpha=0.6,
                    )
                    # Visualization for prediction
                    visualization_image = draw_segmentation_masks(
                        visualization_image,
                        outputs[0, i + 1].sigmoid() > 0.5,
                        colors=colors[i],
                        alpha=0.3,
                    )

                iteration = (epoch + 1) * train_dataset_length
                self.writer_test.add_image(
                    tag="images",
                    img_tensor=visualization_image,
                    global_step=iteration,
                )
                break


def create_cardiac_dataloader_traintest(
    path: str,
    path2: str,
    batch_size: int,
    seed: int = 12345678,
) -> Tuple[DataLoader, DataLoader]:
    global_dataset = CardiacDatasetHDF5(data_path=path, data_path2=path2)
    SPLIT_PERCENTAGE = 0.8

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        global_dataset,
        [SPLIT_PERCENTAGE, 1 - SPLIT_PERCENTAGE],
        generator,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
    )
    return train_dataloader, test_dataloader


random_generator = torch.Generator().manual_seed(1234)
MIN_MAX_CHOICE = torch.tensor([64, 128, 256], dtype=torch.float)


def train_collate_fn(data):
    idx = torch.multinomial(
        input=MIN_MAX_CHOICE,
        num_samples=1,
        replacement=True,
        generator=random_generator,
    )
    current_size = MIN_MAX_CHOICE[idx].int().item()
    images = []
    labels = []
    for x in data:
        image, label = x
        image = image
        label = label
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
    path: str, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextOCRDataset(path, True)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=train_collate_fn,
    )
    test_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=test_collate_fn,
    )
    return train_dataloader, test_dataloader


# from torchvision.io import write_png

# train, test = create_textocr_dataloader("data/textocr", 1)
# preprocessor = v2.Compose(
#     [
#         ToNormalized(),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# for x, y in test:
#     modify, y = preprocessor(x, y)
#     # y = (y * 255).to(torch.uint8)
#     # write_png(x[0], "test.png")

#     print(y[0, 0].min(), y[0, 0].max())
#     visualization_image = draw_segmentation_masks(
#         x[0],
#         y[0, 0] > 0.5,
#         colors=(128, 128, 128),
#         alpha=1.0,
#     )

#     write_png(visualization_image, "test2.png")
#     break
