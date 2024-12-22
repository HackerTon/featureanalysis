import math
import time
from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.experiment.he_experiment import HeExperiment
from src.loss import dice_index, total_loss
from src.experiment.he_experiment import overall_loss
from src.service.base_trainer import BaseTrainer
from src.service.parameter import Parameter


class Trainer(BaseTrainer):
    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
        torch.manual_seed(99951499)

    def run_trainer(self):
        experiment = HeExperiment(parameter=self.parameter)

        train_dataloader = experiment["train_dataloader"]
        test_dataloader = experiment["test_dataloader"]
        model = experiment["model"]
        scheduler = experiment["scheduler"]
        preprocessor = experiment["preprocessor"]
        optimizer = experiment["optimizer"]

        self.train(
            epochs=self.parameter.epoch,
            model=model,
            dataloader_train=train_dataloader,
            dataloader_test=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=total_loss,
            preprocess=preprocessor,
            device=self.parameter.device,
        )

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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
                scheduler=scheduler,
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
            self._save(model=model, epoch=epoch)

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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        rate_to_print = max(
            math.floor(len(dataloader) * self.parameter.train_report_rate), 1
        )
        running_loss = 0.0
        running_iou = 0.0

        lambda_t = torch.tensor((epoch / 100) * 0.02)
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for index, data in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        ):
            with torch.autocast(device_type=device, dtype=dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = preprocess(inputs, labels)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                iou_score = dice_index(outputs.sigmoid(), labels)

                # Dirichlet
                # loss = loss_fn(outputs, labels, lambda_t)
                # alpha = outputs.relu() + 1
                # diriclet_strength = alpha.sum(dim=1, keepdim=True)
                # prediction = alpha / diriclet_strength
                # iou_score = dice_index(prediction, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            if scheduler is not None:
                scheduler.step()
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
        preprocess: v2.Compose,
        loss_fn,
        device: str,
        train_dataset_length: int,
        dtype,
    ):
        sum_loss = 0.0
        sum_iou = 0.0
        lambda_t = torch.tensor((epoch / 100) * 0.02)

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

                    # Dirichlet
                    # loss = loss_fn(outputs, labels, lambda_t)
                    # alpha = outputs.relu() + 1
                    # diriclet_strength = alpha.sum(dim=1, keepdim=True)
                    # prediction = alpha / diriclet_strength
                    # iou_score = dice_index(prediction, labels)

                    sum_loss += loss.item()
                    sum_iou += iou_score.item()

        iteration = (epoch + 1) * train_dataset_length
        avg_loss = sum_loss / len(dataloader)
        avg_iou = sum_iou / len(dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)
        self.writer_test.add_scalar("iou_score", avg_iou, iteration)

    # def _visualize_one_epoch(
    #     self,
    #     epoch: int,
    #     model: torch.nn.Module,
    #     dataloader: DataLoader,
    #     device: Union[torch.device, str],
    #     preprocess: v2.Compose,
    #     train_dataset_length: int,
    # ):
    #     with torch.no_grad():
    #         for data in dataloader:
    #             inputs: torch.Tensor
    #             labels: torch.Tensor
    #             inputs, labels = data

    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             original_image = inputs
    #             inputs, labels = preprocess(inputs, labels)

    #             outputs = model(inputs)
    #             # colors = [
    #             #     (0, 0, 128),
    #             #     (128, 64, 128),
    #             #     (0, 128, 0),
    #             #     (0, 128, 128),
    #             #     (128, 0, 64),
    #             #     (192, 0, 192),
    #             #     (128, 0, 0),
    #             # ]

    #             visualization_image = generate_visualization(
    #                 original_image=original_image,
    #                 prediction=outputs,
    #                 target=labels,
    #             )

    #             # visualization_image = original_image[0]
    #             # for i in range(outputs.size(1) - 1):
    #             #     # Visualization for label
    #             #     visualization_image = draw_segmentation_masks(
    #             #         visualization_image,
    #             #         labels[0, i + 1] > 0.5,
    #             #         colors=colors[i],
    #             #         alpha=0.6,
    #             #     )
    #             #     # Visualization for prediction
    #             #     visualization_image = draw_segmentation_masks(
    #             #         visualization_image,
    #             #         outputs[0, i + 1].sigmoid() > 0.5,
    #             #         colors=colors[i],
    #             #         alpha=0.3,
    #             #     )

    #             iteration = (epoch + 1) * train_dataset_length
    #             self.writer_test.add_image(
    #                 tag="images",
    #                 img_tensor=visualization_image,
    #                 global_step=iteration,
    #             )
    #             break
