import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import crop, resize


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


class RandomResize(torch.nn.Module):
    def __init__(self, output_size=[512, 512]):
        super().__init__()
        self.output_size = output_size

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        i, j, h, w = v2.RandomCrop.get_params(image, output_size=self.output_size)
        return crop(image, i, j, h, w), crop(label, i, j, h, w)


class Resize(torch.nn.Module):
    def __init__(self, output_size=[512, 512]):
        super().__init__()
        self.output_size = output_size

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        return image, resize(label, self.output_size)
