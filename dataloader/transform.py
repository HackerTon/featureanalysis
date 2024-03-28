import torch


class ToNormalized(torch.nn.Module):
    def forward(self, image: torch.Tensor, label: torch.Tensor):
        return image.float() / 255, label.float() / 255
