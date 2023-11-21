from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize

from dataloader.dataloader import UAVIDDataset
from model.model import UNETNetwork
from service.model_saver_service import ModelSaverService


class Trainer:
    def __init__(self, train_report_rate=1000) -> None:
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("log/training/train_{}".format(timestamp))
        self.model_saver = ModelSaverService(path=Path("data/model"), topk=2)
        self.train_report_rate = train_report_rate

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
        device="cpu",
    ):
        running_loss = 0.0

        # Move model to specified device
        model = model.to(device)

        for i, data in enumerate(dataloader):
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

            if i % self.train_report_rate == (self.train_report_rate - 1):
                last_loss = running_loss / self.train_report_rate
                current_training_sample = epoch * len(dataloader) + i + 1
                self.writer.add_scalar("Loss/train", last_loss, current_training_sample)
                running_loss = 0.0

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        mode="cpu",
    ):
        pass

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess,
        device="cpu",
    ):
        for epoch in range(epochs):
            self._train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                preprocess=preprocess,
                device=device,
            )
            self._save(model=model, epoch=epoch)

    def run_trainer(self, device):
        training_data = UAVIDDataset(path="data/processed_dataset/", is_train=True)
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        model = UNETNetwork(numberClass=8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        def dice_loss(pred: torch.Tensor, target: torch.Tensor):
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            nominator = 2 * torch.mul(pred_flat, target_flat)
            denominator = torch.add(pred_flat, target_flat)
            return 1 - torch.mean((nominator + 1e-9) / (denominator + 1e-9))

        def total_loss(pred: torch.Tensor, target: torch.Tensor):
            return torch.nn.functional.cross_entropy(pred, target) + dice_loss(
                pred.softmax(1), target
            )
        preprocess = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.train(
            epochs=5,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=total_loss,
            preprocess=preprocess,
            device=device,
        )
