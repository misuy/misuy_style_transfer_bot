import torch
import torchvision

model = torchvision.models.vgg16(pretrained=True)

torch.save(model, 'models/vgg16/vgg16')