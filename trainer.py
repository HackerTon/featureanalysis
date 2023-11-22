import argparse
import datetime
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from dataloader.dataloader import UAVIDDataset
from model.model import UNETNetwork
from service.model_saver_service import ModelSaverService


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon=1e-9,
):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    nominator = 2 * torch.matmul(pred_flat, target_flat)
    denominator = torch.sum(pred_flat) + torch.sum(target_flat)
    return 1 - ((nominator + epsilon) / (denominator + epsilon))


def total_loss(pred: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.cross_entropy(
        pred,
        target,
    ) + dice_loss(
        pred.softmax(1),
        target,
    )


def train(device: str, batch_size: int, path: str):
    if not Path(path).exists():
        print(f"Dataset not found in '{path}'")
        return

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
    model = UNETNetwork(numberClass=8).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    normalize = normalize.to(device)
    model_saver = ModelSaverService(path=Path("data/savedmodel"))
    timestamp = datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")
    writer = SummaryWriter("data/training/train_{}".format(timestamp))

    model.train(True)
    number_samples = len(train_dataloader)
    percentile_10_split = number_samples // 10
    for epoch in range(80):
        running_loss = 0.0
        for idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = normalize(inputs)
            outputs = model(inputs)
            loss = total_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if idx % percentile_10_split == percentile_10_split - 1:
                current_training_sample = epoch * len(train_dataloader) + idx + 1
                writer.add_scalar(
                    "Loss/train",
                    running_loss / percentile_10_split,
                    current_training_sample,
                )
                print(f"Loss: {running_loss / percentile_10_split}")
                running_loss = 0.0

        model_saver.save(model=model, epoch=epoch)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-bs", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)

    parsed_data = parser.parse_args()
    train(
        device=parsed_data.mode,
        batch_size=parsed_data.batchsize,
        path=parsed_data.path,
    )
