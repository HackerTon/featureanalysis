import torch
from torchvision.transforms import v2


class ToNormalized(torch.nn.Module):
    def forward(self, image: torch.Tensor, label: torch.Tensor):
        return image.float() / 255, label.float() / 255


class ImagenetNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        return self.normalizer(image), label
