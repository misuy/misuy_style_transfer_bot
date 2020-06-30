# misuy_style_transfer_bot
Это репозиторий моего заключительного проекта 1-ой части курса deep learning school.

От меня требовалось содать телеграм бота, умеющего переносить стиль с одной картинки на другую, и опционально с помощью гана накналадывать на картинку определенный стиль (в моем случае имелся в виду кубизм).

Для написания функционала бота я использовал библиотеку aiogram, но он получился не асинхронным, т.к. не смог разобраться в асинхронности.

Для написания моделей использовался фреймворк pytorch. Основу для реальзации NST я взял с официального сайта: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html. Реализацию CycleGan взял из своего прошлого домашнего задания в dlschool: https://github.com/misuy/misuy_cycle_gan, для его создания я вдохновлялся статьей: https://arxiv.org/abs/1703.10593, и реализацией: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. Для этого проекта обучил ган на датасете состоящем из реальных фотографий: https://www.kaggle.com/hsankesara/flickr-image-dataset, и 1500 картин в стиле кубизм из яндекс картинок, которые я спарсил.

# Для тех, кто хочет пользоваться ботом
1)Найдите бота в телеграме @misuy_style_transfer_bot;

2)Отправьте команду /start для начала работы;

Отправьте команду /help, или нажмите на кнопку 'Помощь' на inline клавиатуре, если вам нужна помощь;

Вы можете выбрать 1 из 2-ух режимов работы бота:

1. NST (Можно вызвать командой /nst, или кнопкой NST) - переносит стиль со 2-ой отправленной картинки на 1-ую (картинки нужно отправлять в разных сообщениях).

2. GAN (Можно вызвать командой /gan, или кнопкой GAN) - накладывает определенный стиль на отправленную картинку (в нашем случае это кубизм).

# Для тех, кто хочет клонировать репозиторий
1)Github не клонирует пустые папки, поэтому вам необходимо создать папку images в главной директории проекта [/images] (там где файл main.py), и директорию vgg16 в models [/models/vgg16];

2)Создать файл сonfig.py в главной директории проекта [/config.py], с содержанием: TOKEN='{Ваш токен от бота в телеграме}';

3)Установить библиотеки из requirements.txt командой [python3.7 -m pip install -r requirements.txt], и torch+torchvision командой [python3.7 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html] (torch только для сpu);

4)Запустить файл /models/download_vgg16.py для загрузки предобученной vgg16 для NST;

5)Запустить файл /main.py;

6)Бот запущен!
