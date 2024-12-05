import torch
from datetime import datetime
from src.service.parameter import Parameter
from torch.utils.tensorboard.writer import SummaryWriter
from src.service.model_saver_service import ModelSaverService
from pathlib import Path


class BaseTrainer:
    def __init__(self, parameter: Parameter):
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        directory_name = "data/model/{}_{}".format(
            timestamp,
            parameter.name.replace(" ", "_"),
        )

        self.parameter = parameter
        self.writer_train = SummaryWriter(f"{directory_name}/train")
        self.writer_test = SummaryWriter(f"{directory_name}/test")
        self.model_saver = ModelSaverService(
            path=Path(f"{directory_name}"),
            topk=2,
            name=parameter.name,
        )

    def run_trainer(self):
        raise NotImplementedError("Please implement run_trainer")

    def _save(self, model: torch.nn.Module, epoch: int):
        self.model_saver.save_without_shape(model, epoch)
