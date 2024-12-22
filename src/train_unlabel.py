import argparse
from pathlib import Path

from src.service.parameter import Parameter
from src.service.trainer import Trainer


def run(
    name: str,
    epoch: int,
    device: str,
    batch_size: int,
    path: str,
    learning_rate: float,
    step: int,
):
    if not Path(path).exists():
        print(f"Dataset not found in '{path}'")
        return

    parameter = Parameter()
    parameter.epoch = epoch
    parameter.name = name
    parameter.batch_size_test = batch_size
    parameter.batch_size_train = batch_size
    parameter.learning_rate = learning_rate
    parameter.data_path = path
    parameter.step = step
    parameter.train_report_rate = 0.1
    parameter.device = device

    trainer = Trainer(parameter)
    trainer.run_trainer()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=50, type=int)
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-b", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=0,
        help="Step in your experiment",
    )

    parsed_data = parser.parse_args()
    run(
        name=parsed_data.name,
        epoch=parsed_data.epoch,
        device=parsed_data.mode,
        batch_size=parsed_data.batchsize,
        path=parsed_data.path,
        learning_rate=parsed_data.learning_rate,
        step=parsed_data.step,
    )
