import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import random_split, BatchSampler, RandomSampler, Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize
from torchvision.utils import draw_segmentation_masks

from dataloader.dataloader import CardiacDatasetHDF5, LungDataset
from loss import dice_index, total_loss
from model.model import BackboneType, MultiNet
from service.hyperparamater import Hyperparameter
from service.model_saver_service import ModelSaverService
from utils.utils import combine_channels


class Trainer:
    def __init__(self, train_report_rate: float = 0.50) -> None:
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
        if experiment_num == 5:
            # Initialization
            train_dataloader, test_dataloader = create_cardiac_dataloader_traintest(
                path=hyperparameter.data_path,
                path2=hyperparameter.data_path2,
                batch_size=hyperparameter.batch_size_train,
            )
            model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
            preprocess = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # Move weights to specified device
            model = model.to(device)
            preprocess = preprocess.to(device)

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
                preprocess=preprocess,
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
        preprocess,
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

            # Switch optimizer to SGD
            if epoch == 10:
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

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
        preprocess,
        device: str,
        dtype,
    ):
        rate_to_print = math.floor(len(dataloader) * self.train_report_rate)
        running_loss = 0.0
        running_iou = 0.0
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for index, data in enumerate(dataloader):
            with torch.autocast(device_type=device, dtype=dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs = inputs.to(device).float() / 255
                labels = labels.to(device).float() / 255
                inputs = preprocess(inputs)

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

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        preprocess,
        loss_fn,
        device: str,
        train_dataset_length: int,
        dtype,
    ):
        sum_loss = 0.0
        sum_iou = 0.0

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype):
                for data in dataloader:
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs = inputs.to(device).float() / 255
                    labels = labels.to(device).float() / 255
                    inputs = preprocess(inputs)

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
        preprocess,
        train_dataset_length: int,
    ):
        with torch.no_grad():
            for data in dataloader:
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                original_image = inputs
                inputs = inputs.to(device).float() / 255
                inputs = preprocess(inputs)

                outputs = model(inputs)
                colors = [
                    (0, 0, 0),
                    (0, 0, 128),
                    (128, 64, 128),
                    (0, 128, 0),
                    (0, 128, 128),
                    (128, 0, 64),
                    (192, 0, 192),
                    (128, 0, 0),
                ]

                visualization_image = original_image[0]
                for i in range(1, outputs.size(1)):
                    visualization_image = draw_segmentation_masks(
                        visualization_image,
                        labels[0, i] > 127,
                        colors=colors[i + 2],
                        alpha=0.5,
                    )
                    visualization_image = draw_segmentation_masks(
                        visualization_image,
                        outputs[0, i] > 0.5,
                        colors=colors[i],
                        alpha=0.5,
                    )

                iteration = (epoch + 1) * train_dataset_length
                self.writer_test.add_image(
                    tag="images",
                    img_tensor=visualization_image,
                    global_step=iteration,
                )
                break


def create_train_dataloader(path: str, batch_size: int) -> DataLoader:
    training_data = LungDataset(directory=path, is_train=True)
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    return train_dataloader


def create_test_dataloader(path: str, batch_size: int) -> DataLoader:
    test_data = LungDataset(directory=path, is_train=False)
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
    )
    return test_dataloader


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


# train_dataloader, test_dataloader = create_cardiac_dataloader_traintest(path='/Volumes/storage', path2='/Volumes/storage', batch_size=32)

# # print(len(train_dataset), len(test_dataset))

# # train_dataloader = DataLoader(
# #     global_dataset,
# #     shuffle=True,
# #     batch_size=1,
# #     num_workers=0,
# #     pin_memory=True,
# # )

# # for idx, (x, y) in enumerate(train_dataset):
# #     break

# # initial_time = time.time()
# # for idx, (x, y) in enumerate(global_dataset):
# #     if idx == 1000:
# #         print(x.shape)
# #         break
# # print(time.time() - initial_time)


# initial_time = time.time()
# for idx, (x, y) in enumerate(train_dataloader):
#     if idx == 1000:
#         print(x.shape)
#         break
# print(time.time() - initial_time)
