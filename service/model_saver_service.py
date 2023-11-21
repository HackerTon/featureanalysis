import torch
from pathlib import Path
from os import remove


class ModelSaverService:
    def __init__(self, path: Path, topk: int = 2) -> None:
        self.model_directory = path
        self.topk = topk
        self.latest_model = []

        if not self.model_directory.exists():
            self.model_directory.mkdir(parents=True)

    def _generate_save_name(self, epoch: int):
        return f"{epoch}_model.pt"

    def save(self, model: torch.nn.Module, epoch: int):
        if len(self.latest_model) > self.topk:
            first_epoch_to_delete = self.latest_model.pop(0)
            model_to_delete = self.model_directory.joinpath(
                self._generate_save_name(first_epoch_to_delete)
            )
            remove(model_to_delete)
            print(f"{self._generate_save_name(first_epoch_to_delete)} removed!")

        torch.save(
            model.state_dict(),
            self.model_directory.joinpath(self._generate_save_name(epoch)),
        )
        self.latest_model.append(epoch)
