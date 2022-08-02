from enum import Enum

import torch
from PIL import Image
from torchvision import models
from torchvision import transforms as T

from .labels import imagenet


class Classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device).eval()
        self.model_name = "resnet50"
        self.__transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self.__transforms(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> str:
        label_index = self.model(self.transform(image)).squeeze().argmax().item()
        label = imagenet[label_index]
        return label
