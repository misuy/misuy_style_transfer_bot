import torch
import torchvision
from models.NST_model import NSTModel
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
    content_image = load_image('images/id_{0}/nst/content_image.jpg'.format(user_id))
    style_image = load_image('images/id_{0}/nst/style_image.jpg'.format(user_id))
    image = content_image.clone()

    cnn = torch.load('models/vgg16/vgg16')
    model = NSTModel(cnn=cnn, image=image, style_image=style_image, content_image=content_image, device=device)
    output = transform_image(model.make_image(10))
    output.save('images/id_{0}/nst/output_image.jpg'.format(user_id))
    with open('images/id_{0}/nst/state.txt'.format(user_id), 'w') as f:
        f.write('success')
except:
    with open('images/id_{0}/nst/state.txt'.format(user_id), 'w') as f:
        f.write('error')