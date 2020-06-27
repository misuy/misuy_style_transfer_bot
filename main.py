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
from models.cycle_gan_model import CycleGan, Generator, Discriminator, ResnetBlock
import io
import subprocess




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256


def run_gan_executor(user_id, img_size, device):
    subprocess.Popen(['python3.7', 'gan_executor.py', user_id, img_size, device])
    ret = None
    while True:
        with open('images/id_{0}/gan/state.txt'.format(user_id), 'r') as f:
            data = f.read()
            if data != 'none':
                if data == 'success':
                    ret = True
                    break
                elif data == 'error':
                    ret = False
                    break

    with open('images/id_{0}/gan/state.txt'.format(user_id), 'w') as f:
        f.write('none')
    return(ret)


def run_nst_executor(user_id, img_size, device):
    subprocess.Popen(['python3.7', 'nst_executor.py', user_id, img_size, device])
    ret = None
    while True:
        with open('images/id_{0}/nst/state.txt'.format(user_id), 'r') as f:
            data = f.read()
            if data != 'none':
                if data == 'success':
                    ret = True
                    break
                elif data == 'error':
                    ret = False
                    break
                
    with open('images/id_{0}/nst/state.txt'.format(user_id), 'w') as f:
        f.write('none')
    return(ret)




with open('users_dict.txt', 'r') as f:
    users_dict = json.loads(f.read())

logging.basicConfig(level=logging.INFO)

bot = aiogram.Bot(token=TOKEN)
dp = aiogram.Dispatcher(bot)

inline_help_btn = aiogram.types.InlineKeyboardButton('Помощь', callback_data='help_c')
inline_nst_btn = aiogram.types.InlineKeyboardButton('NST', callback_data='nst_c')
inline_gan_btn = aiogram.types.InlineKeyboardButton('gan_image2kubizm', callback_data='gan_c')
inline_start_kb = aiogram.types.InlineKeyboardMarkup().add(inline_help_btn, inline_nst_btn, inline_gan_btn)

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
    await message.answer('Привет! Я бот, который работает с картинками. Для более подробной информации нажми на кнопку Помощь', reply_markup=inline_start_kb)


@dp.message_handler(commands=['help'])
async def help_def(message: aiogram.types.Message):
    await message.answer('')


@dp.callback_query_handler(lambda c: c.data == 'help_c')
async def help_def_c(callback_query: aiogram.types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, 'ksfgjdsmfj')


@dp.message_handler(commands=['nst'])
async def nst_run_def(message: aiogram.types.Message):
    user_id = message.from_user.id
    users_dict[user_id]['nst'] = {'content_image_path': None, 'style_image_path': None, 'request_id': users_dict[user_id]['requests_count']}
    users_dict[user_id]['gan'] = None
    users_dict[user_id]['requests_count'] += 1
    await message.answer('Отлично! Теперь отправьте 2 картинки в разных сообщениях. С 1-ой возьмется контент, а со 2-ой стиль.')


@dp.callback_query_handler(lambda c: c.data == 'nst_c')
async def nst_run_def_c(callback_query: aiogram.types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    user_id = callback_query.from_user.id
    users_dict[user_id]['nst'] = {'content_image_path': None, 'style_image_path': None, 'request_id': users_dict[user_id]['requests_count']}
    users_dict[user_id]['gan'] = None
    users_dict[user_id]['requests_count'] += 1
    await bot.send_message(user_id, 'Отлично! Теперь отправьте 2 картинки в разных сообщениях. С 1-ой возьмется контент, а со 2-ой стиль.')


@dp.message_handler(commands=['gan'])
async def gan_run_def(message: aiogram.types.Message):
    user_id = message.from_user.id
    users_dict[user_id]['gan'] = {'image_path': None, 'request_id': users_dict[user_id]['requests_count']}
    users_dict[user_id]['nst'] = None
    users_dict[user_id]['requests_count'] += 1
    await message.answer('Отлично! Теперь отправьте картинку.')


@dp.callback_query_handler(lambda c: c.data == 'gan_c')
async def gan_run_def_c(callback_query: aiogram.types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    user_id = callback_query.from_user.id
    users_dict[user_id]['gan'] = {'image_path': None, 'request_id': users_dict[user_id]['requests_count']}
    users_dict[user_id]['nst'] = None
    users_dict[user_id]['requests_count'] += 1
    await bot.send_message(user_id, 'Отлично! Теперь отправьте картинку.')


@dp.message_handler(content_types=['photo'])
async def get_image(message: aiogram.types.Message):
    user_id = message.from_user.id
    file_info = await bot.get_file(message.photo[-1].file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    if users_dict[user_id]['gan'] != None:
        gan = CycleGan(IMG_SIZE, device, mode='load', path='models/kubizm_cycle_gan')
        users_dict[user_id]['gan']['image_path'] = 'images/id_{0}/gan/image.jpg'.format(user_id)
        with open(users_dict[user_id]['gan']['image_path'], 'wb') as f:
            f.write(downloaded_file.read())

        media = aiogram.types.MediaGroup()
        content_f = open(users_dict[user_id]['gan']['image_path'], 'rb')
        media.attach_photo(content_f, 'Контент')

        await message.answer('Для запроса № {0} полученно изображение, начинаю работу.'.format(users_dict[user_id]['gan']['request_id']))
        
        input_image = load_image(users_dict[user_id]['gan']['image_path'])
        output_image = transform_image(gan.make_image(input_image))
        output_image.save('images/id_{0}/gan/output_image.jpg'.format(user_id))

        output_f = open('images/id_{0}/gan/output_image.jpg'.format(user_id), 'rb')
        media.attach_photo(output_f, 'Результат')

        await message.reply_media_group(media=media, reply=False)
        content_f.close()
        output_f.close()
        await message.answer(text='Работа над запросом № {0} закончена!'.format(users_dict[user_id]['gan']['request_id']), reply_markup=inline_start_kb)
        users_dict[user_id]['gan'] = None

        del gan


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
            cnn = torch.load('models/vgg16/vgg16')
            model = NSTModel(cnn=cnn, image=image, style_image=style_image, content_image=content_image, device=device)
            output = transform_image(model.make_image(10))
            output.save('images/id_{0}/nst/output_image.jpg'.format(user_id))
            
            output_f = open('images/id_{0}/nst/output_image.jpg'.format(user_id), 'rb')
            media.attach_photo(output_f, 'Результат')

            await message.reply_media_group(media=media, reply=False)
            content_f.close()
            style_f.close()
            output_f.close()
            await message.answer(text='Работа над запросом № {0} закончена!'.format(users_dict[user_id]['nst']['request_id']), reply_markup=inline_start_kb)
            users_dict[user_id]['nst'] = None

            del cnn
            del model
    else:
        await message.answer('Для начала, выберите один из режимов работы!')
            


if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)
