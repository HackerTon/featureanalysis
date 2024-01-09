from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize

from dataloader.dataloader import UAVIDDataset
from loss import dice_index, dice_index_per_channel, total_loss
from model.model import UNETNetwork, FPNNetwork
from service.hyperparamater import Hyperparameter
from service.model_saver_service import ModelSaverService
from utils.utils import combine_channels, visualize


class Trainer:
    def __init__(self, train_report_rate=1000) -> None:
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.writer_train = SummaryWriter(
            "data/log/training/train_{}".format(timestamp)
        )
        self.writer_test = SummaryWriter("data/log/training/test_{}".format(timestamp))
        self.model_saver = ModelSaverService(path=Path("data/model"), topk=2)
        self.train_report_rate = train_report_rate

    def run_trainer(self, device, hyperparameter: Hyperparameter):
        # Initialization
        train_dataloader = create_train_dataloader(
            path=hyperparameter.data_path,
            batch_size=hyperparameter.batch_size_train,
        )
        test_dataloader = create_test_dataloader(
            path=hyperparameter.data_path,
            batch_size=hyperparameter.batch_size_test,
        )
        model = FPNNetwork(numberClass=8)
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=hyperparameter.learning_rate,
        )
        preprocess = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Move weights to specified device
        model = model.to(device)
        preprocess = preprocess.to(device)

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

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess,
        device=torch.device("cpu"),
    ):
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}")
            self._train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                preprocess=preprocess,
                device=device,
            )
            self._eval_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader_test,
                loss_fn=loss_fn,
                preprocess=preprocess,
                device=device,
                train_dataset_length=len(dataloader_train),
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
        device: torch.device,
    ):
        running_loss = 0.0
        for index, data in enumerate(dataloader):
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data

            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = preprocess(inputs)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if index % self.train_report_rate == (self.train_report_rate - 1):
                last_loss = running_loss / self.train_report_rate
                current_training_sample = epoch * len(dataloader) + index + 1
                self.writer_train.add_scalar(
                    "loss",
                    last_loss,
                    current_training_sample,
                )
                running_loss = 0.0

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        preprocess,
        loss_fn,
        device: torch.device,
        train_dataset_length: int,
    ):
        sum_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = preprocess(inputs)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                sum_loss += loss.item()

        iteration = (epoch + 1) * train_dataset_length
        avg_loss = sum_loss / len(dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)

    def _visualize_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        preprocess,
        train_dataset_length: int,
    ):
        with torch.no_grad():
            for data in dataloader:
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = preprocess(inputs)
                outputs = model(inputs).cpu()
                colors = torch.tensor(
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
                    dtype=torch.uint8,
                )
                grouth_truth_image = combine_channels(labels[0], colors, False)
                grouth_truth_image = grouth_truth_image[..., [2, 1, 0]].cpu()
                predicted_image = combine_channels(outputs[0], colors, True)
                predicted_image = predicted_image[..., [2, 1, 0]].cpu()
                input_image = torch.permute(data[0][0], [1, 2, 0])

                iteration = (epoch + 1) * train_dataset_length
                self.writer_test.add_images(
                    tag="images",
                    img_tensor=torch.stack(
                        [input_image, grouth_truth_image, predicted_image]
                    ),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                break


def create_train_dataloader(path: str, batch_size: int) -> DataLoader:
    training_data = UAVIDDataset(
        path=path,
        is_train=True,
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    return train_dataloader


def create_test_dataloader(path: str, batch_size: int) -> DataLoader:
    test_data = UAVIDDataset(
        path=path,
        is_train=False,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
    )
    return test_dataloader
