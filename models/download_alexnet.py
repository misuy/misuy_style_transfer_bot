import torch
import torchvision

model = torchvision.models.alexnet(pretrained=True)

torch.save(model, 'models/alexnet/alexnet')