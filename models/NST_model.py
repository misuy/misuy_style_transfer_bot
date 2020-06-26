import torch
import torchvision
import numpy as np
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import PIL
import copy


class ContentLoss(torch.nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input_):
        self.loss = torch.nn.functional.mse_loss(input_, self.target)
        return(input_)


def make_gram_matrix(input_):
    a, b, c, d = input_.size()
    features = input_.view(a * b, c * d)
    gram_matrix = features @ features.t()
    return gram_matrix.div(a * b * c * d)


class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_gram_matrix = make_gram_matrix(target_feature).detach()
    
    def forward(self, input_):
        input_gram_matrix = make_gram_matrix(input_)
        self.loss = torch.nn.functional.mse_loss(input_gram_matrix, self.target_gram_matrix)
        return(input_)


class Normalization(torch.nn.Module):

    def __init__(self, device, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])):
        mean = mean.to(device)
        std = std.to(device)
        super(Normalization, self).__init__()
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, image):
        return((image - self.mean) / self.std)


class NSTModel():
    def __init__(self, cnn, image, style_image, content_image, device, style_loss_layers=[1, 2, 3, 4], content_loss_layers=[3]):

        self.image = image
        self.style_image = style_image
        self.content_image = content_image

        max_layer_number = max(max(style_loss_layers), max(content_loss_layers))
        make_names_list = lambda m: list('conv_{0}'.format(el) for el in m)
        style_loss_layers_names = make_names_list(style_loss_layers)
        content_loss_layers_names = make_names_list(content_loss_layers)

        self.content_losses = []
        self.style_losses = []

        cnn = copy.deepcopy(cnn.features.to(device).eval())

        normalization = Normalization(device)

        self.model = torch.nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, torch.nn.Conv2d):
                name = 'conv_{0}'.format(i)
                i += 1
            elif isinstance(layer, torch.nn.ReLU):
                name = 'relu_{0}'.format(i)
                layer = torch.nn.ReLU(inplace=False)
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer')
            
            self.model.add_module(name, layer)

            if name in style_loss_layers_names:
                style_loss = StyleLoss(self.model(self.style_image).detach())
                self.model.add_module('style_loss_{0}'.format(i-1), style_loss)
                self.style_losses.append(style_loss)
            if name in content_loss_layers_names:
                content_loss = ContentLoss(self.model(self.content_image).detach())
                self.model.add_module('content_loss_{0}'.format(i-1), content_loss)
                self.content_losses.append(content_loss)
            
            if i-1 == max_layer_number:
                break
                
        self.model.to(device)
    


    def make_image(self, num_steps, style_weight=1000000, content_weight=1):

        self.optimizer = torch.optim.LBFGS([self.image.requires_grad_()])
        
        for step in range(num_steps):
            def closure():
                self.image.data.clamp_(0, 1)

                self.optimizer.zero_grad()
                self.model(self.image)

                style_loss_sum = 0
                content_loss_sum = 0

                for style_loss in self.style_losses:
                    style_loss_sum += style_loss.loss
                for content_loss in self.content_losses:
                    content_loss_sum += content_loss.loss
                style_loss_sum *= style_weight
                content_loss_sum *= content_weight

                loss = style_loss_sum + content_loss_sum
                loss.backward()

                return(style_loss_sum + content_loss_sum)
            
            self.optimizer.step(closure)
            if step % 10 == 0:
                print('{0}/{1}'.format(step, num_steps))

        self.image.data.clamp_(0, 1)
        return(self.image)