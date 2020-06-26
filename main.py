#from models.NST_model import NSTModel
import torch
import torchvision
import numpy as np
import skimage
import skimage.io
from skimage.io import imsave
import skimage.transform
import matplotlib.pyplot as plt
import PIL
import copy
import aiogram
from config import TOKEN
import logging
import json
import os
from models.NST_model import NSTModel
import io



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256


def load_image(path):
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                torchvision.transforms.ToTensor()                                    
    ])

    image = transform(PIL.Image.open(path)).unsqueeze(0)
    return(image.to(device, torch.float))

def transform_image(image):
    transform = torchvision.transforms.ToPILImage()
    return(transform(image.cpu().clone().squeeze(0)))

cnn = torchvision.models.vgg19(pretrained=True)

'''
style_image = load_image('images/NST_images/picasso.jpg')
content_image = load_image('images/NST_images/dancing.jpg')
image = content_image.clone()

model = NSTModel(cnn=cnn, image=image, style_image=style_image, content_image=content_image, device=device)

output = model.make_image(300)

imsave('images/NST_images/result.jpg', skimage.img_as_ubyte(transform_image(output)))
'''

with open('user_dict.txt', 'r') as f:
    users_dict = json.loads(f.read())

logging.basicConfig(level=logging.INFO)

bot = aiogram.Bot(token=TOKEN)
dp = aiogram.Dispatcher(bot)



@dp.message_handler(commands=['start'])
async def start_def(message: aiogram.types.Message):
    user_id = message.from_user.id
    if user_id not in users_dict.keys():
        try:
            os.mkdir('images/id_{0}'.format(user_id))
            os.mkdir('images/id_{0}/nst'.format(user_id))
            os.mkdir('images/id_{0}/gan'.format(user_id))
        except:
            pass
        users_dict[user_id] = {'nst': None, 'gan': None, 'requests_count': 0}
    await message.answer('Привет! Я бот, который работает с картинками. Для более подробной информации пиши /help')


@dp.message_handler(commands=['help'])
async def help_def(message: aiogram.types.Message):
    await message.answer('')


@dp.message_handler(commands=['nst'])
async def nst_run_def(message: aiogram.types.Message):
    user_id = message.from_user.id
    users_dict[user_id]['nst'] = {'content_image_path': None, 'style_image_path': None, 'request_id': users_dict[user_id]['requests_count']}
    users_dict[user_id]['gan'] = None
    users_dict[user_id]['requests_count'] += 1
    await message.answer('Отлично! Теперь отправьте 2 картинки в разных сообщениях. С 1-ой возьмется контент, а со 2-ой стиль.')


@dp.message_handler(content_types=['photo'])
async def get_image(message: aiogram.types.Message):
    user_id = message.from_user.id
    file_info = await bot.get_file(message.photo[-1].file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    if users_dict[user_id]['gan'] != None:
        pass
    elif users_dict[user_id]['nst'] != None:
        if users_dict[user_id]['nst']['content_image_path'] == None:
            users_dict[user_id]['nst']['content_image_path'] = 'images/id_{0}/nst/content_image.jpg'.format(user_id)

            with open(users_dict[user_id]['nst']['content_image_path'], 'wb') as f:
                f.write(downloaded_file.read())
            await message.answer('Для запроса № {0} полученно изображение контента, теперь отправьте изображение стиля.'.format(users_dict[user_id]['nst']['request_id']))
        else:
            users_dict[user_id]['nst']['style_image_path'] = 'images/id_{0}/nst/style_image.jpg'.format(user_id)
            with open(users_dict[user_id]['nst']['style_image_path'], 'wb') as f:
                f.write(downloaded_file.read())
            media = aiogram.types.MediaGroup()
            content_f = open(users_dict[user_id]['nst']['content_image_path'], 'rb')
            media.attach_photo(content_f, 'Контент')
            style_f = open(users_dict[user_id]['nst']['style_image_path'], 'rb')
            media.attach_photo(style_f, 'Стиль')

            await message.answer('Для запроса № {0} полученно изображение стиля, начинаю работу.'.format(users_dict[user_id]['nst']['request_id']))
            content_image = load_image(users_dict[user_id]['nst']['content_image_path'])
            style_image = load_image(users_dict[user_id]['nst']['style_image_path'])
            image = content_image.clone()
            print('Start work with request {0}'.format(users_dict[user_id]['nst']['request_id']))
            model = NSTModel(cnn=cnn, image=image, style_image=style_image, content_image=content_image, device=device)
            output = transform_image(model.make_image(10))

            output.save('images/id_{0}/nst/output_image.jpg'.format(user_id))
            output_f = open('images/id_{0}/nst/output_image.jpg'.format(user_id), 'rb')
            media.attach_photo(output_f, 'Результат')
            await message.answer('Работа над запросом № {0} закончена!'.format(users_dict[user_id]['nst']['request_id']))
            users_dict[user_id]['nst'] = None
            await message.reply_media_group(media=media, reply=False)
            content_f.close()
            style_f.close()
            output_f.close()
    else:
        await message.answer('для начала, выберите один из режимов работы!')
            


if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)
