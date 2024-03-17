import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize

from dataloader.dataloader import CardiacDataset, LungDataset
from loss import dice_index, total_loss
from model.model import BackboneType, FPNNetwork, MultiNet, UNETNetwork
from service.hyperparamater import Hyperparameter
from service.model_saver_service import ModelSaverService
from utils.utils import combine_channels


class Trainer:
    def __init__(self, train_report_rate=1000) -> None:
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.writer_train = SummaryWriter(
            "data/log/training/train_{}".format(timestamp)
        )
        self.writer_test = SummaryWriter("data/log/training/test_{}".format(timestamp))
        self.model_saver = ModelSaverService(path=Path("data/model"), topk=2)
        self.train_report_rate = train_report_rate

    def run_trainer(
        self, device: str, hyperparameter: Hyperparameter, experiment_num: int
    ):
        if experiment_num == 1:
            # Initialization
            train_dataloader = create_train_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            test_dataloader = create_test_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_test,
            )
            model = UNETNetwork(numberClass=2)
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
        elif experiment_num == 2:
            # Initialization
            train_dataloader = create_train_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            test_dataloader = create_test_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_test,
            )
            model = FPNNetwork(numberClass=2)
            optimizer = torch.optim.SGD(
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
        elif experiment_num == 3:
            # Initialization
            train_dataloader = create_train_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            test_dataloader = create_test_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_test,
            )
            model = MultiNet(numberClass=2, backboneType=BackboneType.RESNET34)
            optimizer = torch.optim.SGD(
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
                dataloader_test=train_dataloader,
                optimizer=optimizer,
                loss_fn=total_loss,
                preprocess=preprocess,
                device=device,
            )
        elif experiment_num == 4:
            # Initialization
            train_dataloader = create_train_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            test_dataloader = create_test_dataloader(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_test,
            )
            model = MultiNet(numberClass=2, backboneType=BackboneType.RESNET50)
            optimizer = torch.optim.SGD(
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
        elif experiment_num == 5:
            # Initialization
            train_dataloader = create_cardiac_dataloader_train(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
            # test_dataloader = create_test_dataloader(
            #     path=hyperparameter.data_path,
            #     batch_size=hyperparameter.batch_size_test,
            # )
            model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
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
                dataloader_test=None,
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

            # Unfreze on second epoch
            if epoch == 1:
                # Unfreeze backbone
                model: FPNNetwork = model
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

            if dataloader_test is None:
                self._visualize_one_epoch(
                    epoch=epoch,
                    model=model,
                    dataloader=dataloader_train,
                    preprocess=preprocess,
                    train_dataset_length=len(dataloader_train),
                    device=device,
                )
            else:
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
        running_loss = 0.0
        running_iou = 0.0
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for index, data in enumerate(dataloader):
            with torch.autocast(device_type=device, dtype=dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = preprocess(inputs)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                iou_score = dice_index(outputs.softmax(1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_iou += iou_score.item()

            if index % self.train_report_rate == (self.train_report_rate - 1):
                current_training_sample = epoch * len(dataloader) + index + 1
                self.writer_train.add_scalar(
                    "loss",
                    running_loss / self.train_report_rate,
                    current_training_sample,
                )
                self.writer_train.add_scalar(
                    "iou_score",
                    running_iou / self.train_report_rate,
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
            for data in dataloader:
                with torch.autocast(device_type=device, dtype=dtype):
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs = preprocess(inputs)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    iou_score = dice_index(outputs.softmax(1), labels)

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

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = preprocess(inputs)
                outputs = model(inputs).cpu()
                colors = torch.tensor(
                    [
                        [0, 0, 0],
                        [0, 0, 128],
                        [128, 64, 128],
                        # [0, 128, 0],
                        # [0, 128, 128],
                        # [128, 0, 64],
                        # [192, 0, 192],
                        # [128, 0, 0],
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
                        [input_image, grouth_truth_image / 255, predicted_image / 255]
                    ),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                break


def create_train_dataloader(path: str, batch_size: int) -> DataLoader:
    training_data = LungDataset(directory=path, is_train=True)
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader


def create_test_dataloader(path: str, batch_size: int) -> DataLoader:
    test_data = LungDataset(directory=path, is_train=False)
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return test_dataloader


def create_cardiac_dataloader_train(path: str, batch_size: int) -> DataLoader:
    training_data = CardiacDataset(directory_path=path, is_train=True)
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader


def create_cardiac_dataloader_test(path: str, batch_size: int) -> DataLoader:
    test_data = CardiacDataset(directory_path=path, is_train=False)
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return test_dataloader


# from torchvision.io import write_png
# training_data = CardiacDataset(directory_path='data/cardiac', is_train=True)
# train_dataloader = DataLoader(
#     training_data,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True,
# )
# for x, y in train_dataloader:
#     print(x.shape, y.shape)
#     write_png((y[0, 1] * 255).to(torch.uint8).unsqueeze(0), 'test.png')
#     break
