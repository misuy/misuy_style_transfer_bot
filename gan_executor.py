import torch
import torchvision
from models.cycle_gan_model import ResnetBlock, Generator, Discriminator, CycleGan
from sys import argv
import PIL

user_id, IMG_SIZE, device = argv[1], int(argv[2]), argv[3]

def load_image(path):
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                torchvision.transforms.ToTensor()                                    
    ])

    image = transform(PIL.Image.open(path)).unsqueeze(0)
    return(image.to(device, torch.float))

def transform_image(image):
    transform = torchvision.transforms.ToPILImage()
    return(transform(image.cpu().clone().squeeze(0)))

try:
    gan = CycleGan(IMG_SIZE, device, mode='load', path='models/kubizm_cycle_gan')

    input_image = load_image('images/id_{0}/gan/image.jpg'.format(user_id))
    output_image = transform_image(gan.make_image(input_image))
    output_image.save('images/id_{0}/gan/output_image.jpg'.format(user_id))
    with open('images/id_{0}/gan/state.txt'.format(user_id), 'w') as f:
        f.write('success')
except:
    with open('images/id_{0}/gan/state.txt'.format(user_id), 'w') as f:
        f.write('error')