import argparse
from pathlib import Path

import torch
from torchvision.io.image import write_png
from src.model.model import UNETNetwork
from torch.utils.data import DataLoader
from src.dataloader.dataset.he_dataset import HeDataset
from src.dataloader.transform import ToNormalized, ImagenetNormalize
from torchvision.transforms.v2 import Compose
from tqdm import tqdm

BATCH_SIZE = 128
UNCERTAINTY_THRESHOLD = 0.01
NSAMPLES = 998  # NSAMPLES for uniform random selection


def write_sample(image, label, directory: Path, index):
    image_path = directory.joinpath(f"{index}_image.png")
    label_path = directory.joinpath(f"{index}_label.png")
    write_png(image, str(image_path))
    write_png(label, str(label_path))


def baseline_inference(
    model,
    dataloader,
    preprocessor,
    pseudo_path,
):
    sample_num = 0
    for image, mask in tqdm(dataloader):
        image_after, mask = preprocessor(image.cuda(), mask.cuda())
        with torch.no_grad():
            prediction = model(image_after)

        # Dirichlet version
        prediction = prediction.softmax(dim=1)

        for index in range(prediction.size(0)):
            if sample_num == NSAMPLES:
                break

            write_sample(
                image[index].cpu(),
                prediction[index].argmax(dim=0, keepdim=True).to(torch.uint8).cpu(),
                pseudo_path,
                sample_num,
            )
            sample_num += 1

        if sample_num == NSAMPLES:
            break


def dirichlet_inference(
    model,
    dataloader,
    preprocessor,
    pseudo_path,
):
    sample_num = 0
    for image, mask in tqdm(dataloader):
        image_after, mask = preprocessor(image.cuda(), mask.cuda())
        with torch.no_grad():
            prediction = model(image_after)

        # Dirichlet version
        alpha = prediction.relu() + 1
        dirichlet_strength = alpha.sum(dim=1)
        prediction = alpha / dirichlet_strength.unsqueeze(1)

        uncertainty_median = (
            (3 / dirichlet_strength).flatten(start_dim=1).median(dim=1).values
        )

        wanted_samples = (uncertainty_median < UNCERTAINTY_THRESHOLD).argwhere().cpu()
        for index in wanted_samples:
            write_sample(
                image[index[0]].cpu(),
                prediction[index[0]].argmax(dim=0, keepdim=True).to(torch.uint8).cpu(),
                pseudo_path,
                sample_num,
            )
            sample_num += 1


def run(namespace):
    pseudo_path = Path(namespace.pseudo_dir)
    if pseudo_path.exists():
        for file in pseudo_path.glob("*"):
            file.unlink()
    else:
        pseudo_path.mkdir()

    model = UNETNetwork(numberClass=3)
    model.load_state_dict(torch.load(namespace.model))
    model.eval()
    model.cuda()
    preprocessor = Compose([ToNormalized(), ImagenetNormalize()])

    dataset = HeDataset(namespace.dataset_dir, "data/temp")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True if namespace.model == "baseline" else False,
        num_workers=4,
    )

    if namespace.mode == "baseline":
        baseline_inference(model, dataloader, preprocessor, pseudo_path)
    elif namespace.mode == "dirichlet":
        dirichlet_inference(model, dataloader, preprocessor, pseudo_path)
    else:
        print("mode is not baseline or dirichlet")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model path with has pt")
    parser.add_argument("--dataset_dir", required=True, help="he dataset")
    parser.add_argument("--pseudo_dir", required=True, help="path to save")
    parser.add_argument("--mode", required=True, help="baseline or dirichlet")
    parsed_data = parser.parse_args()
    run(parsed_data)
