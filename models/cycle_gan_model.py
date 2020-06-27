import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
import numpy as np


class ResnetBlock(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(layer_size, layer_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(layer_size, layer_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(layer_size),
        )
    
    def forward(self, inp):
        return(inp + self.block(inp))




class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        model = []

        n_downsampling = 3
        layer_size = 32


        model += [nn.Conv2d(3, layer_size, kernel_size=7, padding=3)
                  
        ]


        for i in range(n_downsampling):
            prev_layer_size = layer_size
            layer_size *= 2
            model += [nn.Conv2d(prev_layer_size, layer_size, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(layer_size),
                      nn.ReLU(inplace=True)
                                      
            ]
        

        prev_layer_size = layer_size
        n_blocks = 5
        for i in range(n_blocks):
            model += [ResnetBlock(prev_layer_size)]


        n_upsampling = n_downsampling  
        for i in range(n_upsampling):
            layer_size = prev_layer_size // 2
            model += [nn.ConvTranspose2d(prev_layer_size, layer_size, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(layer_size),
                      nn.ReLU(inplace=True),                      
            ]
            prev_layer_size //= 2


        model += [nn.Conv2d(prev_layer_size, 3, kernel_size=7, stride=1, padding=3),
                  nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)



    def forward(self, inp):
        return(self.model(inp))




class Discriminator(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()

        sequence = []
        prev_layer_size = 3
        layer_size = 64

        sequence = [nn.Conv2d(3, layer_size, kernel_size=4, padding=1, stride=2), 
                    nn.LeakyReLU(0.2, inplace=True),
        ]

        for i in range(n_layers - 1):
            prev_layer_size = layer_size
            layer_size *= 2
            sequence += [nn.Conv2d(in_channels=prev_layer_size, out_channels=layer_size, kernel_size=4, padding=1, stride=2),
                         nn.BatchNorm2d(layer_size),
                         nn.LeakyReLU(0.2, inplace=True),
            ]

        prev_layer_size = layer_size
        layer_size *= 2
        sequence += [nn.Conv2d(in_channels=prev_layer_size, out_channels=layer_size, kernel_size=4, padding=1, stride=1),
                     nn.BatchNorm2d(layer_size),
                     nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [nn.Conv2d(in_channels=layer_size, out_channels=1, kernel_size=4, padding=1, stride = 1)]

        self.model = nn.Sequential(*sequence)
    


    def forward(self, input):
        return(self.model(input))




class CycleGan():
    def __init__(self, img_size, device, mode='create', path=None):
        self.device = device
        if mode == 'create':
            self.G_X_Y = Generator().to(self.device)
            self.D_X_Y = Discriminator().to(self.device)
            self.G_Y_X = Generator().to(self.device)
            self.D_Y_X = Discriminator().to(self.device)
        elif mode == 'load':
            self.load_models(path)
        else:
            print('Mode name is not correct')
            raise(NameError)

        self.BCEloss = nn.BCEWithLogitsLoss()
        self.L1Loss = nn.L1Loss()


    
    def save_models(self, path):
        torch.save(self.G_X_Y, path + '/G_X_Y')
        torch.save(self.G_Y_X, path + '/G_Y_X')
        torch.save(self.D_X_Y, path + '/D_X_Y')
        torch.save(self.D_Y_X, path + '/D_Y_X')
    


    def load_models(self, path):
        self.G_X_Y = torch.load(path + '/G_X_Y', map_location=self.device)
        self.G_X_Y.train()
        self.G_Y_X = torch.load(path + '/G_Y_X', map_location=self.device)
        self.G_Y_X.train()
        self.D_X_Y = torch.load(path + '/D_X_Y', map_location=self.device)
        self.D_X_Y.train()
        self.D_Y_X = torch.load(path + '/D_Y_X', map_location=self.device)
        self.D_Y_X.train()
        


    def set_requires_grad(self, models, value):
        for model in models:
            for param in model.parameters():
                param.requires_grad = value



    def gan_criterion(self, prediction, is_true):
        if is_true:
            target_tensor = torch.Tensor(1.0).expand_as(prediction)
        else:
            target_tensor = torch.Tensor(0.0).expand_as(prediction)

        loss = self.BCEloss(prediction.cpu(), target_tensor.cpu())

        return(loss.to(self.device))



    def discriminators_backward(self, discriminator_X_Y, discriminator_Y_X, fake_X, fake_Y, real_X, real_Y):
        real_D_X_Y_loss = self.gan_criterion(discriminator_X_Y(real_Y), True)
        fake_D_X_Y_loss = self.gan_criterion(discriminator_X_Y(fake_Y.detach()), False)
        self.loss_D_X_Y = real_D_X_Y_loss + fake_D_X_Y_loss
        self.loss_D_X_Y.backward()

        real_D_Y_X_loss = self.gan_criterion(discriminator_Y_X(real_X), True)
        fake_D_Y_X_loss = self.gan_criterion(discriminator_Y_X(fake_X.detach()), False)
        self.loss_D_Y_X = real_D_Y_X_loss + fake_D_Y_X_loss
        self.loss_D_Y_X.backward()
        return(real_D_X_Y_loss, fake_D_X_Y_loss)



    def generators_backward(self, generator_X_Y, discriminator_X_Y, generator_Y_X, discriminator_Y_X, fake_X, fake_Y, real_X, real_Y, comp_X, comp_Y, lmb):
        gan_loss_X_Y = self.gan_criterion(discriminator_X_Y(fake_Y), True)
        gan_loss_Y_X = self.gan_criterion(discriminator_Y_X(fake_X), True)

        cycle_loss_X_Y = self.L1Loss(comp_X, real_X) * lmb
        cycle_loss_Y_X = self.L1Loss(comp_Y, real_Y) * lmb

        #identy_loss_X_Y = self.L1Loss(generator_X_Y(real_Y), real_Y) * lmb
        #identy_loss_Y_X = self.L1Loss(generator_Y_X(real_X), real_X) * lmb

        self.G_loss = gan_loss_X_Y + gan_loss_Y_X + cycle_loss_X_Y + cycle_loss_Y_X
        self.G_loss.backward()



    def train(self, data_loader, epochs, lr, lmb, test_set):
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_X_Y.parameters(), self.G_Y_X.parameters()), lr=lr)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_X_Y.parameters(), self.D_Y_X.parameters()), lr=lr)
        test_set = test_set.to(self.device)
        show_rate = epochs // 100
        counter = 0


        for epoch in range(epochs):
            real_loss_list = []
            fake_loss_list = []
            for X_batch, Y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                self.fake_Y = self.G_X_Y(X_batch)  # G(X)
                self.comp_X = self.G_Y_X(self.fake_Y)   # F(G(X))
                self.fake_X = self.G_Y_X(Y_batch)  # F(Y)
                self.comp_Y = self.G_X_Y(self.fake_X)   # G(F(Y))
                # generator
                self.set_requires_grad([self.D_X_Y, self.D_Y_X], False)
                self.optimizer_G.zero_grad()
                self.generators_backward(self.G_X_Y, self.D_X_Y, self.G_Y_X, self.D_Y_X, self.fake_X, self.fake_Y, X_batch, Y_batch, self.comp_X, self.comp_Y, lmb)
                self.optimizer_G.step()
                # discriminator
                self.set_requires_grad([self.D_X_Y, self.D_Y_X], True)
                self.optimizer_D.zero_grad()
                real_loss, fake_loss = self.discriminators_backward(self.D_X_Y, self.D_Y_X, self.fake_X, self.fake_Y, X_batch, Y_batch)
                real_loss_list.append(real_loss.detach().cpu().item())
                fake_loss_list.append(fake_loss.detach().cpu().item())
                self.optimizer_D.step()


            if counter % show_rate == 0:
                print('{0}/{1}'.format(counter + 1, epochs))
                print(sum(real_loss_list) / len(real_loss_list), sum(fake_loss_list) / len(fake_loss_list))
                test_set_output = self.G_X_Y(test_set)
                for i in range(4):
                    plt.subplot(2, 4, i+1)
                    plt.imshow(np.rollaxis(test_set[i].detach().cpu().numpy(), 0, 3))
                    plt.axis('off')
                    plt.subplot(2, 4, i+5)
                    plt.imshow(np.rollaxis(test_set_output[i].detach().cpu().numpy(), 0, 3))
                    plt.axis('off')
                plt.show()
                self.save_models('model')


            counter += 1


    def make_image(self, image):
        output = self.G_X_Y(image)
        return(output)