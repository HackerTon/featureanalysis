import argparse
from pathlib import Path

from service.hyperparamater import Hyperparameter
from service.trainer import Trainer


def run(device: str, batch_size: int, path: str):
    if not Path(path).exists():
        print(f"Dataset not found in '{path}'")
        return

    trainer = Trainer(5)
    hyperparameter = Hyperparameter(
        learning_rate=0.001,
        batch_size_test=32,
        batch_size_train=16,
        data_path=path,
    )
    trainer.run_trainer(device=device, hyperparameter=hyperparameter)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-bs", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)

    parsed_data = parser.parse_args()
    run(
        device=parsed_data.mode,
        batch_size=parsed_data.batchsize,
        path=parsed_data.path,
    )
