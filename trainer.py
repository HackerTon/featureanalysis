import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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


def train():
    training_data = UAVIDDataset(
        path="/Users/babi/Programs/high_performance_analysis_system/data/processed_dataset/",
        is_train=True,
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    model = UNETNetwork(numberClass=8).to("mps")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    optimizer_sgd = torch.optim.SGD(params=model.parameters(), lr=0.001)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(
        "mps"
    )

    model_saver = ModelSaverService(path=Path("data/savedmodel"))

    for epoch in range(4):
        model.train(True)
        running_loss = 0.0
        for idx, data in enumerate(tqdm(train_dataloader)):
            initial_time = time.perf_counter()
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data
            optimizer.zero_grad()
            optimizer_sgd.zero_grad()

            inputs = inputs.to("mps")
            labels = labels.to("mps")

            inputs = normalize(inputs)
            outputs = model(inputs)

            loss = total_loss(outputs, labels)
            loss.backward()

            if epoch == 0:
                optimizer_sgd.step()
            else:
                optimizer.step()

            running_loss += loss.item()

            if idx % 10 == 9:
                time_taken = (time.perf_counter() - initial_time) / 10
                print(
                    f"Loss: {running_loss / (idx + 1)}, Avg Time Taken: {time_taken}ms"
                )

        model_saver.save(model=model, epoch=epoch)


if __name__ == "__main__":
    train()
