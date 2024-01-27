import argparse
from pathlib import Path

from service.hyperparamater import Hyperparameter
from service.trainer import Trainer


def run(epoch: int, device: str, batch_size: int, path: str, experiment_num: int, learning_rate: float):
    if not Path(path).exists():
        print(f"Dataset not found in '{path}'")
        return

    trainer = Trainer(train_report_rate=5)
    hyperparameter = Hyperparameter(
        epoch=epoch,
        learning_rate=learning_rate,
        batch_size_test=batch_size * 2,
        batch_size_train=batch_size,
        data_path=path,
    )
    trainer.run_trainer(
        device=device,
        hyperparameter=hyperparameter,
        experiment_num=experiment_num,
    )


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=50, type=int)
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-bs", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-l", "--learning_rate", default=0.001, type=str)
    parser.add_argument(
        "-x",
        "--experiment",
        required=True,
        type=int,
        help="Experiment number. Refer to run_trainer function",
    )

    parsed_data = parser.parse_args()
    run(
        epoch=parsed_data.epoch,
        device=parsed_data.mode,
        batch_size=parsed_data.batchsize,
        path=parsed_data.path,
        experiment_num=parsed_data.experiment,
        learning_rate=parsed_data.learning_rate
    )
