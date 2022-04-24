
import torch
import torch.nn as nn
from torchvision import models

def VGG16():

    model = models.vgg16_bn()
    model.classifier = nn.Sequential(
    nn.Linear(in_features=512, out_features=100, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=100, out_features=10, bias=True)
    )

    return model